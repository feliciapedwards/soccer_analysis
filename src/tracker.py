from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from ultralytics import YOLO

from .detector import Detection, FrameDetections

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
    occlusion/re-entry), each excess ID is remapped to the spatially nearest
    existing capped ID so the output never exceeds `max_players` unique IDs.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence: float = 0.4,
        device: str = "",
        max_players: int = 25,
    ):
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.device = device or None
        self.max_players = max_players

        # raw ByteTrack ID → capped output ID
        self._id_map: Dict[int, int] = {}
        # capped ID → last known center (for proximity remapping)
        self._capped_centers: Dict[int, Tuple[float, float]] = {}
        self._next_capped_id: int = 1

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

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])

                if cls == _PERSON_CLASS:
                    raw_id = int(ids[i]) if ids is not None else -1
                    capped_id = self._resolve_id(raw_id, (x1, y1, x2, y2))
                    players.append(
                        TrackedPlayer(player_id=capped_id, bbox=(x1, y1, x2, y2), confidence=conf)
                    )
                elif cls == _BALL_CLASS:
                    det = Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls)
                    if ball is None or conf > ball.confidence:
                        ball = det

        # Update last-known centers for capped IDs
        for p in players:
            self._capped_centers[p.player_id] = p.center

        return TrackedFrame(frame_index=frame_index, players=players, ball=ball)

    def _resolve_id(self, raw_id: int, bbox: Tuple[float, float, float, float]) -> int:
        """Map a raw ByteTrack ID to a capped output ID."""
        if raw_id in self._id_map:
            return self._id_map[raw_id]

        if self._next_capped_id <= self.max_players:
            # Capacity available — assign new capped ID
            capped = self._next_capped_id
            self._next_capped_id += 1
        else:
            # At capacity — merge to spatially nearest existing capped ID
            center = _center_of(bbox)
            capped = min(
                self._capped_centers,
                key=lambda cid: _dist(center, self._capped_centers[cid]),
            )

        self._id_map[raw_id] = capped
        return capped
