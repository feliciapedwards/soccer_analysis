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
    possession_player_id: int = -1   # who has the ball at this moment (-1 = in transit)
    possession_team_id: int = -1     # that player's team


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
        self._prev_player_aspects: Dict[int, float] = {}  # player_id -> bbox aspect ratio (w/h)
        self._player_grounded_frames: Dict[int, int] = {}  # player_id -> consecutive grounded frames
        self._foul_emitted_players: set = set()   # player_ids that already had a FOUL emitted (cooldown)
        self._player_was_standing: Dict[int, bool] = {}  # player was upright before going down

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
            poss_id = self._current_possessor if self._current_possessor is not None else -1
            poss_team = team_ids.get(poss_id, -1) if poss_id != -1 else -1
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
                possession_player_id=poss_id,
                possession_team_id=poss_team,
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

        # --- FOUL detection: player on the ground (bbox becomes wide/flat) ---
        # A standing player has aspect ratio (w/h) < 1 (tall bbox).
        # A player on the ground has aspect ratio > foul_ground_aspect_ratio (wide/flat bbox).
        # We require foul_min_frames consecutive "grounded" frames before emitting.
        for player in players:
            x1, y1, x2, y2 = player.bbox
            w = x2 - x1
            h = y2 - y1 if (y2 - y1) > 0 else 1
            aspect = w / h  # > 1 means wide (grounded), < 1 means tall (standing)

            prev_aspect = self._prev_player_aspects.get(player.player_id, aspect)
            is_grounded = aspect >= self.cfg.foul_ground_aspect_ratio
            was_standing = self._player_was_standing.get(player.player_id, False)

            if is_grounded and was_standing and player.player_id not in self._foul_emitted_players:
                self._player_grounded_frames[player.player_id] = (
                    self._player_grounded_frames.get(player.player_id, 0) + 1
                )
                if self._player_grounded_frames[player.player_id] == self.cfg.foul_min_frames:
                    team = team_ids.get(player.player_id, -1)
                    events.append(Event(
                        frame_number=frame, timestamp_sec=timestamp_sec,
                        player_id=player.player_id, team_id=team,
                        event_type="FOUL",
                        player_x=player.center[0], player_y=player.center[1],
                        ball_x=ball_center[0] if ball_center else None,
                        ball_y=ball_center[1] if ball_center else None,
                        ball_speed=self._ball_speed,
                        event_confidence=self.cfg.foul_confidence,
                    ))
                    self._foul_emitted_players.add(player.player_id)
            else:
                # Player is standing — reset grounded counter and cooldown
                self._player_grounded_frames.pop(player.player_id, None)
                self._foul_emitted_players.discard(player.player_id)
                # Mark as standing only if clearly upright (aspect well below threshold)
                if aspect < self.cfg.foul_ground_aspect_ratio * 0.7:
                    self._player_was_standing[player.player_id] = True

            self._prev_player_aspects[player.player_id] = aspect

        # Update previous state
        self._prev_ball_center = ball_center
        for p in players:
            prev = self._prev_player_centers.get(p.player_id, p.center)
            self._prev_player_speeds[p.player_id] = _dist(p.center, prev)
            self._prev_player_centers[p.player_id] = p.center

        return events
