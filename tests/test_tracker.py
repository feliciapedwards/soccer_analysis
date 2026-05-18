import unittest
from unittest.mock import patch

import numpy as np

from src.tracker import Tracker


class _FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
        self.conf = np.array([0.9, 0.85], dtype=float)
        self.cls = np.array([0, 0], dtype=float)
        self.id = None

    def __len__(self):
        return len(self.xyxy)


class _FakeYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def track(self, *args, **kwargs):
        return [type("Result", (), {"boxes": _FakeBoxes()})()]


class TrackerTests(unittest.TestCase):
    @patch("src.tracker.YOLO", _FakeYOLO)
    def test_untracked_detections_do_not_collapse_to_one_id(self):
        tracker = Tracker(model_name="fake.pt", reid_enabled=False)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        tracked = tracker.track_frame(0, frame)

        self.assertEqual(len(tracked.players), 2)
        self.assertNotEqual(tracked.players[0].player_id, tracked.players[1].player_id)


if __name__ == "__main__":
    unittest.main()
