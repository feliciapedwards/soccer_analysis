import cv2
from pathlib import Path
from typing import Generator, Tuple
import numpy as np


def iter_frames(video_path: str) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_sec, frame_bgr) for every frame in the video.

    Handles iPhone video rotation via the capture's orientation property.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotation_code = _get_rotation_code(cap)

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
            timestamp_sec = frame_index / fps
            yield frame_index, timestamp_sec, frame
            frame_index += 1
    finally:
        cap.release()


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _get_rotation_code(cap: cv2.VideoCapture):
    """Return a cv2.ROTATE_* constant if the video needs rotation, else None."""
    # OpenCV 4.5+ exposes this property for some backends
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    except Exception:
        return None

    rotation_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return rotation_map.get(rotation)
