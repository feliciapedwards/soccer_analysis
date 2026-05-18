"""Train a jersey color classifier from labeled crops.

Usage:
    python scripts/train_jersey_classifier.py

Expects labeled crops in:
    data/crops/red/       ← red shirt players
    data/crops/white/     ← white shirt players
    data/crops/referee/   ← referee (yellow shirt)

Saves trained model to:
    models/jersey_classifier.pkl

The pipeline will automatically use this model for team classification
the next time it runs.
"""

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CROPS_DIR = Path("data/crops")
MODEL_PATH = Path("models/jersey_classifier.pkl")

# Class label mapping — must match team_classifier.py LABEL_MAP
CLASSES = {
    "red": 0,       # red team → green box
    "white": 1,     # white team → blue box
    "referee": 2,   # referee → gray box (stored as 2, returned as -1)
}

# Crop region — must match team_classifier.py constants
_CROP_TOP = 0.15
_CROP_BOTTOM = 0.40


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """Extract a combined color feature vector from a jersey crop image.

    Features:
      - Pixel class fractions: white, red, yellow (3 values)
      - 16-bin hue histogram on all pixels (16 values)
      - Mean saturation and value (2 values)
    Total: 21 features
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    total = max(len(H), 1)

    white  = float(np.sum((S < 60) & (V > 150))) / total
    red    = float(np.sum(((H < 10) | (H > 160)) & (S > 100) & (V > 80))) / total
    yellow = float(np.sum((H >= 15) & (H <= 40) & (S > 100) & (V > 100))) / total

    h_hist, _ = np.histogram(H, bins=16, range=(0, 180))
    h_hist = h_hist.astype(np.float32) / max(h_hist.sum(), 1)

    mean_s = float(S.mean()) / 255.0
    mean_v = float(V.mean()) / 255.0

    return np.concatenate([[white, red, yellow], h_hist, [mean_s, mean_v]])


def load_labeled_crops() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, paths = [], [], []

    for class_name, label in CLASSES.items():
        class_dir = CROPS_DIR / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found — skipping '{class_name}' class")
            continue

        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not imgs:
            print(f"  Warning: no images in {class_dir}")
            continue

        print(f"  {class_name}: {len(imgs)} images")
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)
            paths.append(str(img_path))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def main():
    print("Loading labeled crops...")
    X, y, paths = load_labeled_crops()

    if len(X) == 0:
        print("\nNo labeled data found.")
        print("Run first: python scripts/extract_crops.py --video data/test_clip.mp4")
        print("Then sort crops into data/crops/red/, data/crops/white/, data/crops/referee/")
        sys.exit(1)

    unique, counts = np.unique(y, return_counts=True)
    label_names = {v: k for k, v in CLASSES.items()}
    print(f"\nClass distribution:")
    for lbl, cnt in zip(unique, counts):
        print(f"  {label_names[lbl]}: {cnt} samples")

    if len(unique) < 2:
        print("\nNeed at least 2 classes to train. Add more labeled data.")
        sys.exit(1)

    if len(X) < 10:
        print(f"\nOnly {len(X)} samples — need at least 10. Add more labeled crops.")
        sys.exit(1)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])

    # Cross-validation
    n_splits = min(5, min(counts))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"\nCross-val accuracy: {scores.mean():.1%} ± {scores.std():.1%}")
    else:
        print("\nNot enough samples per class for cross-validation — training on all data.")

    clf.fit(X, y)

    # Full training report
    y_pred = clf.predict(X)
    class_names = [label_names[i] for i in sorted(label_names)]
    print("\nTraining classification report:")
    print(classification_report(y, y_pred, target_names=class_names))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "classes": CLASSES, "extract_features": "v1"}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("\nRe-run the pipeline — it will automatically use the trained model.")


if __name__ == "__main__":
    main()
