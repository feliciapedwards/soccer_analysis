import unittest

from src.config import Config
from src.detector import Detection
from src.event_detector import EventDetector
from src.tracker import TrackedFrame, TrackedPlayer


class EventDetectorTests(unittest.TestCase):
    def test_ball_position_uses_current_frame_possessor(self):
        detector = EventDetector(Config(possession_distance_px=30))
        tracked = TrackedFrame(
            frame_index=0,
            players=[TrackedPlayer(player_id=7, bbox=(40, 40, 60, 80), confidence=0.9)],
            ball=Detection(bbox=(48, 58, 52, 62), confidence=0.95, class_id=32),
        )

        events = detector.process_frame(tracked, team_ids={7: 1}, timestamp_sec=0.0)
        ball_position = next(event for event in events if event.event_type == "BALL_POSITION")

        self.assertEqual(ball_position.player_id, 7)
        self.assertEqual(ball_position.team_id, 1)

    def test_goal_and_out_of_bounds_are_edge_triggered(self):
        detector = EventDetector(
            Config(
                goal_left=[0, 0, 30, 30],
                pitch_boundary=[10, 10, 90, 90],
            )
        )
        tracked = TrackedFrame(
            frame_index=0,
            players=[],
            ball=Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=32),
        )

        first_events = detector.process_frame(tracked, team_ids={}, timestamp_sec=0.0)
        second_events = detector.process_frame(
            TrackedFrame(frame_index=1, players=[], ball=tracked.ball),
            team_ids={},
            timestamp_sec=1 / 30,
        )

        self.assertEqual([event.event_type for event in first_events].count("GOAL"), 1)
        self.assertEqual([event.event_type for event in first_events].count("OUT_OF_BOUNDS"), 1)
        self.assertNotIn("GOAL", [event.event_type for event in second_events])
        self.assertNotIn("OUT_OF_BOUNDS", [event.event_type for event in second_events])


if __name__ == "__main__":
    unittest.main()
