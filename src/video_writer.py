from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .tracker import TrackedFrame
from .event_detector import Event
from .detector import Detection

# Events prominent enough to show as on-screen overlays
_OVERLAY_EVENTS = {"SHOT", "GOAL", "FOUL", "PASS", "OUT_OF_BOUNDS"}

# BGR colors for overlay text labels
_EVENT_COLORS = {
    "GOAL": (0, 215, 255),
    "SHOT": (0, 165, 255),
    "FOUL": (0, 0, 255),
    "PASS": (255, 255, 0),
    "OUT_OF_BOUNDS": (200, 200, 200),
}

_BALL_COLOR = (0, 0, 255)       # red
_DEFAULT_PLAYER_COLOR = (180, 180, 180)
_LABEL_BG_ALPHA = 0.45


class VideoWriter:
    """Writes an annotated copy of the input video with bounding boxes, player IDs, and event overlays."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_width: int,
        frame_height: int,
        team_color_0: List[int],
        team_color_1: List[int],
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (frame_width, frame_height)
        )
        self._team_colors = {
            0: tuple(team_color_0),
            1: tuple(team_color_1),
        }
        # Overlay events linger for N frames
        self._overlay_queue: List[Tuple[int, str]] = []  # (expire_frame, label)
        self._linger_frames = max(1, int(fps * 1.5))

    def write_frame(
        self,
        frame_bgr: np.ndarray,
        tracked: TrackedFrame,
        team_ids: Dict[int, int],
        events: List[Event],
    ):
        canvas = frame_bgr.copy()
        frame_idx = tracked.frame_index

        # --- Draw players ---
        for player in tracked.players:
            x1, y1, x2, y2 = [int(v) for v in player.bbox]
            team = team_ids.get(player.player_id, -1)
            color = self._team_colors.get(team, _DEFAULT_PLAYER_COLOR)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"P{player.player_id}"
            _draw_label(canvas, label, x1, y1, color)

        # --- Draw ball ---
        if tracked.ball:
            bx1, by1, bx2, by2 = [int(v) for v in tracked.ball.bbox]
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            cv2.circle(canvas, (cx, cy), 8, _BALL_COLOR, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)

        # --- Queue new overlay events ---
        for event in events:
            if event.event_type in _OVERLAY_EVENTS:
                label = event.event_type
                if event.player_id > 0:
                    label += f" P{event.player_id}"
                self._overlay_queue.append((frame_idx + self._linger_frames, label))

        # Expire old overlays
        self._overlay_queue = [(exp, lbl) for exp, lbl in self._overlay_queue if exp > frame_idx]

        # --- Draw active overlays stacked in top-left ---
        unique_labels = list(dict.fromkeys(lbl for _, lbl in self._overlay_queue))
        for i, label in enumerate(unique_labels[:5]):  # max 5 simultaneous
            event_type = label.split()[0]
            color = _EVENT_COLORS.get(event_type, (255, 255, 255))
            y_pos = 40 + i * 36
            cv2.putText(canvas, label, (14, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(canvas, label, (14, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        # --- Timestamp ---
        h, w = canvas.shape[:2]
        ts = f"{int(tracked.frame_index // 30 // 60):02d}:{int(tracked.frame_index // 30 % 60):02d}"
        cv2.putText(canvas, ts, (w - 90, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(canvas, ts, (w - 90, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        self._writer.write(canvas)

    def release(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int, color: tuple):
    """Draw a small label above the bounding box with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 2
    lx, ly = x, max(y - th - pad * 2, 0)
    cv2.rectangle(canvas, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2), (0, 0, 0), -1)
    cv2.putText(canvas, text, (lx + pad, ly + th + pad), font, scale, color, thickness)
