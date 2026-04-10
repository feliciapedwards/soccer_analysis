import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Config:
    # Detection
    yolo_model: str = "yolov8m.pt"
    yolo_confidence: float = 0.4
    device: str = ""

    # Tracking
    tracker_config: str = "bytetrack.yaml"
    max_players: int = 25
    track_buffer: int = 10
    reid_enabled: bool = True
    reid_threshold: float = 20.0

    # Team classification
    team_kmeans_frames: int = 30

    # Possession
    possession_distance_px: int = 80

    # Pass
    pass_min_frames: int = 3

    # Shot
    shot_speed_threshold_px: float = 25.0
    shot_min_frames: int = 3

    # Goal regions [x_min, y_min, x_max, y_max]
    goal_left: Optional[List[int]] = None
    goal_right: Optional[List[int]] = None

    # Pitch boundary [x_min, y_min, x_max, y_max]
    pitch_boundary: Optional[List[int]] = None

    # Foul detection
    foul_ground_aspect_ratio: float = 1.2
    foul_min_frames: int = 4
    foul_confidence: float = 0.7

    # Video output colors (BGR)
    team_color_0: list = None
    team_color_1: list = None

    def __post_init__(self):
        if self.team_color_0 is None:
            self.team_color_0 = [0, 255, 0]
        if self.team_color_1 is None:
            self.team_color_1 = [255, 80, 0]


def load_config(path: str = "config.yaml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return Config(
        yolo_model=raw.get("yolo_model", "yolov8m.pt"),
        yolo_confidence=raw.get("yolo_confidence", 0.4),
        device=raw.get("device", ""),
        tracker_config=raw.get("tracker_config", "bytetrack.yaml"),
        team_kmeans_frames=raw.get("team_kmeans_frames", 30),
        possession_distance_px=raw.get("possession_distance_px", 80),
        pass_min_frames=raw.get("pass_min_frames", 3),
        shot_speed_threshold_px=raw.get("shot_speed_threshold_px", 25.0),
        shot_min_frames=raw.get("shot_min_frames", 3),
        goal_left=raw.get("goal_left"),
        goal_right=raw.get("goal_right"),
        pitch_boundary=raw.get("pitch_boundary"),
        foul_ground_aspect_ratio=raw.get("foul_ground_aspect_ratio", 1.2),
        foul_min_frames=raw.get("foul_min_frames", 4),
        foul_confidence=raw.get("foul_confidence", 0.7),
        max_players=raw.get("max_players", 25),
        track_buffer=raw.get("track_buffer", 10),
        reid_enabled=raw.get("reid_enabled", True),
        reid_threshold=raw.get("reid_threshold", 20.0),
        team_color_0=raw.get("team_color_0"),
        team_color_1=raw.get("team_color_1"),
    )
