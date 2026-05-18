"""Extract jersey crops from a video for manual labeling.

Usage:
    python scripts/extract_crops.py --video data/test_clip.mp4

After running, sort the saved crops into:
    data/crops/red/       ← players with red shirts
    data/crops/white/     ← players with white shirts
    data/crops/referee/   ← referee (yellow shirt)

Then run:
    python scripts/train_jersey_classifier.py
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO

_PERSON_CLASS = 0
# Skip bboxes taller than this fraction of frame height (bystanders close to camera)
_MAX_BBOX_HEIGHT_FRACTION = 0.60
# Crop: skip head/neck (top 15%), focus on chest (15–40% of bbox height)
_CROP_TOP = 0.15
_CROP_BOTTOM = 0.40


def extract_crops(
    video_path: str,
    output_dir: str = "data/crops/unlabeled",
    every_n_frames: int = 15,
    model_name: str = "yolov8m.pt",
    confidence: float = 0.4,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_bbox_height = frame_height * _MAX_BBOX_HEIGHT_FRACTION

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            results = model(frame, conf=confidence, classes=[_PERSON_CLASS], verbose=False)
            if results and results[0].boxes is not None:
                for i, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    h = y2 - y1

                    # Skip bystanders/large detections near camera
                    if h > max_bbox_height:
                        continue

                    # Tight chest crop — stays well above shorts
                    cy1 = y1 + int(h * _CROP_TOP)
                    cy2 = y1 + int(h * _CROP_BOTTOM)
                    crop = frame[max(cy1, 0):max(cy2, 0), max(x1, 0):max(x2, 0)]

                    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                        continue

                    crop_resized = cv2.resize(crop, (64, 64))
                    fname = f"frame{frame_idx:06d}_det{i:02d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, fname), crop_resized)
                    saved += 1

        frame_idx += 1

    cap.release()
    print(f"\nSaved {saved} crops to {output_dir}/")
    print("\nNext steps:")
    print("  1. Open the crops folder and sort images into subfolders:")
    print("       data/crops/red/       ← red shirts")
    print("       data/crops/white/     ← white shirts")
    print("       data/crops/referee/   ← yellow shirt (referee)")
    print("  2. Run: python scripts/train_jersey_classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract jersey crops for labeling.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", default="data/crops/unlabeled", help="Where to save crops")
    parser.add_argument("--every-n-frames", type=int, default=15, help="Sample every N frames (default: 15)")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=0.4, help="Detection confidence threshold")
    args = parser.parse_args()

    extract_crops(
        video_path=args.video,
        output_dir=args.output_dir,
        every_n_frames=args.every_n_frames,
        model_name=args.model,
        confidence=args.confidence,
    )
