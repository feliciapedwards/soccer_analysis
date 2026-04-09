from collections import defaultdict
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Assigns team_id (0 or 1) to each player using jersey color clustering.

    Strategy:
    - Extract the top-half crop of each player bounding box (jersey region).
    - Convert to HSV and compute a color histogram or mean HSV.
    - After collecting samples across `min_frames` frames, fit KMeans(k=2).
    - Assign each player_id a stable team_id via majority vote.
    """

    def __init__(self, min_frames: int = 30):
        self.min_frames = min_frames
        self._samples: Dict[int, list] = defaultdict(list)  # player_id -> list of HSV feature vectors
        self._team_assignments: Dict[int, int] = {}
        self._kmeans: Optional[KMeans] = None
        self._fitted = False

    def update(self, player_id: int, bbox: Tuple[float, float, float, float], frame_bgr: np.ndarray):
        """Extract jersey color feature from bounding box crop and store sample."""
        feature = _extract_jersey_feature(bbox, frame_bgr)
        if feature is not None:
            self._samples[player_id].append(feature)

    def fit_if_ready(self) -> bool:
        """Fit KMeans once enough samples have been collected. Returns True when fitted."""
        if self._fitted:
            return True

        all_samples_count = sum(len(v) for v in self._samples.values())
        if all_samples_count < self.min_frames * max(len(self._samples), 1):
            return False

        all_features = []
        sample_player_ids = []
        for pid, feats in self._samples.items():
            for f in feats:
                all_features.append(f)
                sample_player_ids.append(pid)

        if len(all_features) < 2:
            return False

        X = np.array(all_features)
        self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = self._kmeans.fit_predict(X)

        # Majority vote per player_id
        votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for pid, label in zip(sample_player_ids, labels):
            votes[pid][label] += 1
        for pid, vote_counts in votes.items():
            self._team_assignments[pid] = max(vote_counts, key=vote_counts.get)

        self._fitted = True
        return True

    def get_team(self, player_id: int, bbox: Tuple[float, float, float, float], frame_bgr: np.ndarray) -> int:
        """Return team_id (0 or 1) for a player. -1 if not yet determined."""
        if not self._fitted:
            return -1

        if player_id in self._team_assignments:
            return self._team_assignments[player_id]

        # New player seen after fitting — classify on the fly
        feature = _extract_jersey_feature(bbox, frame_bgr)
        if feature is None or self._kmeans is None:
            return -1
        label = int(self._kmeans.predict([feature])[0])
        self._team_assignments[player_id] = label
        return label


def _extract_jersey_feature(bbox: Tuple[float, float, float, float], frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Crop the jersey region (top 40% of bbox) and return mean HSV as a feature vector."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    # Use the upper torso region to avoid shorts/socks confusion
    jersey_y2 = y1 + int(h * 0.45)
    crop = frame_bgr[y1:jersey_y2, x1:x2]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Mean H, S, V across the crop
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    return mean_hsv
