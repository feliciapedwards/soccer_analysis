import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.config import load_config
from src.team_classifier import TeamClassifier
from src.video_loader import _rotation_swaps_dimensions


class ConfigAndClassifierTests(unittest.TestCase):
    def test_empty_config_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("")

            cfg = load_config(str(config_path))

        self.assertEqual(cfg.reid_threshold, 0.3)
        self.assertEqual(cfg.track_buffer, 10)

    def test_invalid_box_validation_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("goal_left: [1, 2, 3]\n")

            with self.assertRaises(ValueError):
                load_config(str(config_path))

    def test_team_classifier_respects_min_frames(self):
        classifier = TeamClassifier(min_frames=3)
        frame = np.zeros((40, 40, 3), dtype=np.uint8)

        with patch("src.team_classifier._get_chest_crop", return_value=frame), patch(
            "src.team_classifier._count_jersey_colors_from_crop", return_value=(0, 10, 0)
        ), patch("src.team_classifier._decide_pixel_vote", return_value=1):
            classifier.update(10, (0, 0, 20, 20), frame)
            classifier.update(10, (0, 0, 20, 20), frame)
            self.assertNotIn(10, classifier._team_assignments)

            classifier.update(10, (0, 0, 20, 20), frame)

        self.assertEqual(classifier._team_assignments[10], 1)

    def test_rotation_helper_detects_dimension_swaps(self):
        self.assertTrue(_rotation_swaps_dimensions(cv2.ROTATE_90_CLOCKWISE))
        self.assertFalse(_rotation_swaps_dimensions(None))


if __name__ == "__main__":
    unittest.main()
