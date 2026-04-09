from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

# COCO class IDs
_PERSON_CLASS = 0
_BALL_CLASS = 32


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int  # 0=person, 32=sports ball

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def is_person(self) -> bool:
        return self.class_id == _PERSON_CLASS

    @property
    def is_ball(self) -> bool:
        return self.class_id == _BALL_CLASS


@dataclass
class FrameDetections:
    frame_index: int
    players: List[Detection]
    ball: Optional[Detection]


class Detector:
    def __init__(self, model_name: str = "yolov8m.pt", confidence: float = 0.4, device: str = ""):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device or None  # None = ultralytics auto-selects

    def detect(self, frame_index: int, frame_bgr: np.ndarray) -> FrameDetections:
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            classes=[_PERSON_CLASS, _BALL_CLASS],
            device=self.device,
            verbose=False,
        )
        players: List[Detection] = []
        ball: Optional[Detection] = None

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                det = Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls)
                if det.is_person:
                    players.append(det)
                elif det.is_ball and (ball is None or conf > ball.confidence):
                    ball = det  # keep highest-confidence ball detection

        return FrameDetections(frame_index=frame_index, players=players, ball=ball)
