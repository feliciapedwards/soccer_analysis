from typing import Dict, Optional
import numpy as np


class AppearanceStore:
    """Stores and matches player appearance features for re-identification.

    When a player leaves and re-enters the frame, ByteTrack assigns a new raw ID.
    This class matches that new ID to an existing player by comparing jersey color
    features, recovering the original capped player ID.

    Uses an exponential moving average to keep the stored feature up to date.
    """

    def __init__(self, ema_alpha: float = 0.3, distance_threshold: float = 20.0):
        self._features: Dict[int, np.ndarray] = {}  # capped_id -> HSV feature
        self.ema_alpha = ema_alpha
        self.distance_threshold = distance_threshold

    def update(self, player_id: int, feature: np.ndarray):
        """Update or initialize appearance feature for a capped player ID."""
        if player_id in self._features:
            self._features[player_id] = (
                (1 - self.ema_alpha) * self._features[player_id]
                + self.ema_alpha * feature
            )
        else:
            self._features[player_id] = feature.copy()

    def find_match(self, feature: np.ndarray, exclude_ids: set = None) -> Optional[int]:
        """Find the capped player ID whose stored feature is closest to the given feature.

        Returns None if no match is within the distance threshold.
        """
        if not self._features:
            return None

        candidates = {
            pid: np.linalg.norm(feature - stored)
            for pid, stored in self._features.items()
            if exclude_ids is None or pid not in exclude_ids
        }
        if not candidates:
            return None

        best_id = min(candidates, key=candidates.get)
        best_dist = candidates[best_id]
        return best_id if best_dist <= self.distance_threshold else None
