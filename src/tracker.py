from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math
import numpy as np
from ultralytics import YOLO

from .detector import Detection, FrameDetections
from .appearance_store import AppearanceStore
from .team_classifier import _extract_jersey_feature

_PERSON_CLASS = 0
_BALL_CLASS = 32


def _center_of(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class TrackedPlayer:
    player_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TrackedFrame:
    frame_index: int
    players: List[TrackedPlayer]
    ball: Optional[Detection]  # ball is not tracked by ByteTrack, just detected


class Tracker:
    """Wraps YOLOv8 + ByteTrack to provide persistent player_id across frames.

    ID capping: if ByteTrack assigns more than `max_players` unique IDs (due to
    occlusion/re-entry), each excess ID is first matched by appearance (ReID),
    then by spatial proximity as a fallback.

    track_buffer: number of frames ByteTrack holds a lost track before discarding.
    A larger value reduces ID churn for brief occlusions.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence: float = 0.4,
        device: str = "",
        max_players: int = 25,
        track_buffer: int = 10,
        reid_enabled: bool = True,
        reid_threshold: float = 20.0,
    ):
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.device = device or None
        self.max_players = max_players
        self.track_buffer = track_buffer

        # raw ByteTrack ID → capped output ID
        self._id_map: Dict[int, int] = {}
        # capped ID → last known center (for proximity fallback)
        self._capped_centers: Dict[int, Tuple[float, float]] = {}
        self._next_capped_id: int = 1

        # Appearance-based ReID
        self._reid_enabled = reid_enabled
        self._appearance = AppearanceStore(distance_threshold=reid_threshold) if reid_enabled else None

    def track_frame(self, frame_index: int, frame_bgr: np.ndarray) -> TrackedFrame:
        results = self.model.track(
            frame_bgr,
            conf=self.confidence,
            classes=[_PERSON_CLASS, _BALL_CLASS],
            tracker=self.tracker_config,
            device=self.device,
            persist=True,  # maintain track state across calls
            verbose=False,
        )

        players: List[TrackedPlayer] = []
        ball: Optional[Detection] = None

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id  # may be None if no tracks yet

            # IDs already assigned this frame — prevents two detections claiming same capped ID
            assigned_this_frame: Set[int] = set()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])

                if cls == _PERSON_CLASS:
                    raw_id = int(ids[i]) if ids is not None else -1
                    capped_id = self._resolve_id(raw_id, (x1, y1, x2, y2), frame_bgr, assigned_this_frame)
                    assigned_this_frame.add(capped_id)
                    players.append(
                        TrackedPlayer(player_id=capped_id, bbox=(x1, y1, x2, y2), confidence=conf)
                    )
                elif cls == _BALL_CLASS:
                    det = Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls)
                    if ball is None or conf > ball.confidence:
                        ball = det

        # Update last-known centers and appearance for capped IDs
        for p in players:
            self._capped_centers[p.player_id] = p.center
            if self._appearance is not None:
                feature = _extract_jersey_feature(p.bbox, frame_bgr)
                if feature is not None:
                    self._appearance.update(p.player_id, feature)

        return TrackedFrame(frame_index=frame_index, players=players, ball=ball)

    def _resolve_id(
        self,
        raw_id: int,
        bbox: Tuple[float, float, float, float],
        frame_bgr: np.ndarray,
        assigned_this_frame: Set[int],
    ) -> int:
        """Map a raw ByteTrack ID to a capped output ID.

        Resolution order:
        1. Already seen this raw_id → return cached mapping
        2. Appearance ReID → match to existing player by jersey color
        3. Capacity available → assign new capped ID
        4. At capacity → fall back to spatially nearest existing capped ID
        """
        if raw_id in self._id_map:
            return self._id_map[raw_id]

        # --- Appearance ReID ---
        if self._appearance is not None:
            feature = _extract_jersey_feature(bbox, frame_bgr)
            if feature is not None:
                matched = self._appearance.find_match(feature, exclude_ids=assigned_this_frame)
                if matched is not None:
                    self._id_map[raw_id] = matched
                    return matched

        # --- Assign new capped ID if capacity available ---
        if self._next_capped_id <= self.max_players:
            capped = self._next_capped_id
            self._next_capped_id += 1
        else:
            # At capacity — fall back to nearest spatial neighbor not already used this frame
            center = _center_of(bbox)
            candidates = {
                cid: _dist(center, c)
                for cid, c in self._capped_centers.items()
                if cid not in assigned_this_frame
            }
            capped = min(candidates, key=candidates.get) if candidates else 1

        self._id_map[raw_id] = capped
        return capped
