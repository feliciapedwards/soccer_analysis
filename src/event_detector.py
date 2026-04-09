from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from .config import Config
from .tracker import TrackedFrame, TrackedPlayer
from .detector import Detection


@dataclass
class Event:
    frame_number: int
    timestamp_sec: float
    player_id: int
    team_id: int
    event_type: str
    player_x: Optional[float]
    player_y: Optional[float]
    ball_x: Optional[float]
    ball_y: Optional[float]
    ball_speed: float
    event_confidence: float


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _in_box(point: Tuple[float, float], box: List[int]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


class EventDetector:
    """Rule-based soccer event detector operating on a stream of TrackedFrames."""

    def __init__(self, config: Config):
        self.cfg = config

        # State
        self._prev_ball_center: Optional[Tuple[float, float]] = None
        self._ball_speed: float = 0.0

        self._current_possessor: Optional[int] = None  # player_id
        self._possession_start_frame: int = 0
        self._free_ball_frames: int = 0  # frames since last possession

        self._prev_player_speeds: Dict[int, float] = {}  # player_id -> speed last frame
        self._prev_player_centers: Dict[int, Tuple[float, float]] = {}

        # Temporal filters to reduce noise
        self._shot_fast_frames: int = 0           # consecutive frames with high ball speed
        self._shot_emitted: bool = False           # prevent repeat SHOT for same kick
        self._foul_close_frames: Dict[tuple, int] = {}  # pair -> consecutive close frames

    def process_frame(
        self,
        tracked: TrackedFrame,
        team_ids: Dict[int, int],
        timestamp_sec: float,
    ) -> List[Event]:
        """Process one frame and return any events detected."""
        events: List[Event] = []
        ball = tracked.ball
        players = tracked.players
        frame = tracked.frame_index

        ball_center = _center(ball.bbox) if ball else None

        # --- Ball speed ---
        if ball_center and self._prev_ball_center:
            self._ball_speed = _dist(ball_center, self._prev_ball_center)
        else:
            self._ball_speed = 0.0

        # --- BALL_POSITION event (every frame ball is detected) ---
        if ball_center:
            events.append(Event(
                frame_number=frame,
                timestamp_sec=timestamp_sec,
                player_id=-1,
                team_id=-1,
                event_type="BALL_POSITION",
                player_x=None,
                player_y=None,
                ball_x=ball_center[0],
                ball_y=ball_center[1],
                ball_speed=self._ball_speed,
                event_confidence=ball.confidence if ball else 0.0,
            ))

        # --- OUT_OF_BOUNDS ---
        if ball_center and self.cfg.pitch_boundary:
            if not _in_box(ball_center, self.cfg.pitch_boundary):
                events.append(Event(
                    frame_number=frame, timestamp_sec=timestamp_sec,
                    player_id=-1, team_id=-1,
                    event_type="OUT_OF_BOUNDS",
                    player_x=None, player_y=None,
                    ball_x=ball_center[0], ball_y=ball_center[1],
                    ball_speed=self._ball_speed, event_confidence=0.9,
                ))

        # --- GOAL ---
        if ball_center:
            for goal_box, label in [
                (self.cfg.goal_left, "GOAL"),
                (self.cfg.goal_right, "GOAL"),
            ]:
                if goal_box and _in_box(ball_center, goal_box):
                    events.append(Event(
                        frame_number=frame, timestamp_sec=timestamp_sec,
                        player_id=-1, team_id=-1,
                        event_type="GOAL",
                        player_x=None, player_y=None,
                        ball_x=ball_center[0], ball_y=ball_center[1],
                        ball_speed=self._ball_speed, event_confidence=0.95,
                    ))

        # --- Possession resolution ---
        possessor: Optional[TrackedPlayer] = None
        min_dist = float("inf")

        if ball_center:
            for player in players:
                d = _dist(player.center, ball_center)
                if d < self.cfg.possession_distance_px and d < min_dist:
                    min_dist = d
                    possessor = player

        if possessor:
            pc = possessor.center
            team = team_ids.get(possessor.player_id, -1)

            # --- POSSESSION event (only emit when possessor CHANGES) ---
            if possessor.player_id != self._current_possessor:
                events.append(Event(
                    frame_number=frame, timestamp_sec=timestamp_sec,
                    player_id=possessor.player_id, team_id=team,
                    event_type="POSSESSION",
                    player_x=pc[0], player_y=pc[1],
                    ball_x=ball_center[0] if ball_center else None,
                    ball_y=ball_center[1] if ball_center else None,
                    ball_speed=self._ball_speed, event_confidence=1.0 - (min_dist / self.cfg.possession_distance_px),
                ))

            # --- PASS event ---
            if (
                self._current_possessor is not None
                and self._current_possessor != possessor.player_id
                and self._free_ball_frames >= self.cfg.pass_min_frames
            ):
                events.append(Event(
                    frame_number=frame, timestamp_sec=timestamp_sec,
                    player_id=possessor.player_id, team_id=team,
                    event_type="PASS",
                    player_x=pc[0], player_y=pc[1],
                    ball_x=ball_center[0] if ball_center else None,
                    ball_y=ball_center[1] if ball_center else None,
                    ball_speed=self._ball_speed, event_confidence=0.8,
                ))

            self._current_possessor = possessor.player_id
            self._free_ball_frames = 0
            self._shot_fast_frames = 0
            self._shot_emitted = False
        else:
            self._free_ball_frames += 1

            # --- SHOT event: require `shot_min_frames` consecutive fast frames ---
            if self._ball_speed >= self.cfg.shot_speed_threshold_px and self._current_possessor is not None:
                self._shot_fast_frames += 1
                if self._shot_fast_frames >= self.cfg.shot_min_frames and not self._shot_emitted:
                    prev_possessor_player = next(
                        (p for p in players if p.player_id == self._current_possessor), None
                    )
                    ppx, ppy = (prev_possessor_player.center if prev_possessor_player else (None, None))
                    team = team_ids.get(self._current_possessor, -1)
                    events.append(Event(
                        frame_number=frame, timestamp_sec=timestamp_sec,
                        player_id=self._current_possessor, team_id=team,
                        event_type="SHOT",
                        player_x=ppx, player_y=ppy,
                        ball_x=ball_center[0] if ball_center else None,
                        ball_y=ball_center[1] if ball_center else None,
                        ball_speed=self._ball_speed,
                        event_confidence=min(1.0, self._ball_speed / (self.cfg.shot_speed_threshold_px * 2)),
                    ))
                    self._shot_emitted = True
                    self._current_possessor = None
            else:
                self._shot_fast_frames = 0

        # --- FOUL detection: require `foul_min_frames` consecutive close frames ---
        player_centers = {p.player_id: p.center for p in players}
        player_ids = list(player_centers.keys())
        active_pairs = set()

        for i in range(len(player_ids)):
            for j in range(i + 1, len(player_ids)):
                pid_a, pid_b = player_ids[i], player_ids[j]
                pair = (min(pid_a, pid_b), max(pid_a, pid_b))
                ca, cb = player_centers[pid_a], player_centers[pid_b]
                proximity = _dist(ca, cb)

                if proximity < self.cfg.foul_proximity_px:
                    active_pairs.add(pair)
                    self._foul_close_frames[pair] = self._foul_close_frames.get(pair, 0) + 1

                    if self._foul_close_frames[pair] == self.cfg.foul_min_frames:
                        speed_a = _dist(ca, self._prev_player_centers.get(pid_a, ca))
                        speed_b = _dist(cb, self._prev_player_centers.get(pid_b, cb))
                        prev_speed_a = self._prev_player_speeds.get(pid_a, speed_a)
                        prev_speed_b = self._prev_player_speeds.get(pid_b, speed_b)

                        drop_a = (prev_speed_a - speed_a) / (prev_speed_a + 1e-6)
                        drop_b = (prev_speed_b - speed_b) / (prev_speed_b + 1e-6)

                        if drop_a > self.cfg.foul_speed_drop_threshold or drop_b > self.cfg.foul_speed_drop_threshold:
                            foul_pid = pid_a if drop_a > drop_b else pid_b
                            foul_center = player_centers[foul_pid]
                            team = team_ids.get(foul_pid, -1)
                            events.append(Event(
                                frame_number=frame, timestamp_sec=timestamp_sec,
                                player_id=foul_pid, team_id=team,
                                event_type="FOUL",
                                player_x=foul_center[0], player_y=foul_center[1],
                                ball_x=ball_center[0] if ball_center else None,
                                ball_y=ball_center[1] if ball_center else None,
                                ball_speed=self._ball_speed,
                                event_confidence=self.cfg.foul_confidence,
                            ))

        # Reset counters for pairs no longer close
        for pair in list(self._foul_close_frames):
            if pair not in active_pairs:
                del self._foul_close_frames[pair]

        # Update previous state
        self._prev_ball_center = ball_center
        for p in players:
            prev = self._prev_player_centers.get(p.player_id, p.center)
            self._prev_player_speeds[p.player_id] = _dist(p.center, prev)
            self._prev_player_centers[p.player_id] = p.center

        return events
