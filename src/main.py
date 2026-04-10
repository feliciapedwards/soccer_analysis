#!/usr/bin/env python3
"""Soccer event extraction pipeline.

Usage:
    python src/main.py --video data/game.mp4 --config config.yaml --output output/events.csv
    python src/main.py --video data/game.mp4 --output-video output/annotated.mp4
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python src/main.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.video_loader import iter_frames, get_video_frame_count, get_video_fps
from src.tracker import Tracker
from src.team_classifier import TeamClassifier
from src.event_detector import EventDetector
from src.csv_exporter import CSVExporter
from src.video_writer import VideoWriter

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract soccer events from a video file.")
    parser.add_argument("--video", required=True, help="Path to input video file (.mp4)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--output", default="output/events.csv", help="Path to output CSV (default: output/events.csv)")
    parser.add_argument("--output-video", default=None, help="(Optional) Path to write annotated output video (.mp4)")
    return parser.parse_args()


def run(video_path: str, config_path: str, output_path: str, output_video_path: str = None):
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    print(f"Initializing tracker (model={cfg.yolo_model}, device={cfg.device or 'auto'}, max_players={cfg.max_players})...")
    tracker = Tracker(
        model_name=cfg.yolo_model,
        tracker_config=cfg.tracker_config,
        confidence=cfg.yolo_confidence,
        device=cfg.device,
        max_players=cfg.max_players,
        track_buffer=cfg.track_buffer,
        reid_enabled=cfg.reid_enabled,
        reid_threshold=cfg.reid_threshold,
    )

    team_classifier = TeamClassifier(min_frames=cfg.team_kmeans_frames)
    event_detector = EventDetector(cfg)

    total_frames = get_video_frame_count(video_path)
    fps = get_video_fps(video_path)
    print(f"Processing: {video_path} ({total_frames} frames @ {fps:.1f} fps)")

    # Determine frame dimensions for video writer
    frame_width, frame_height = _get_video_dimensions(video_path)

    video_ctx = (
        VideoWriter(
            output_video_path,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            team_color_0=cfg.team_color_0,
            team_color_1=cfg.team_color_1,
        )
        if output_video_path
        else _NullContext()
    )

    with CSVExporter(output_path) as exporter, video_ctx as vwriter:
        team_fitted = False

        for frame_index, timestamp_sec, frame_bgr in iter_frames(video_path):
            # Track players + ball
            tracked = tracker.track_frame(frame_index, frame_bgr)

            # Update team classifier samples
            for player in tracked.players:
                team_classifier.update(player.player_id, player.bbox, frame_bgr)

            # Try to fit team classifier once enough data is collected
            if not team_fitted:
                team_fitted = team_classifier.fit_if_ready()

            # Build team_id lookup for this frame
            team_ids = {
                p.player_id: team_classifier.get_team(p.player_id, p.bbox, frame_bgr)
                for p in tracked.players
            }

            # Detect events
            events = event_detector.process_frame(tracked, team_ids, timestamp_sec)

            # Write events to CSV
            exporter.write_events(events)

            # Write annotated video frame (if requested)
            if vwriter:
                vwriter.write_frame(frame_bgr, tracked, team_ids, events)

            # Progress reporting every 100 frames
            if frame_index % 100 == 0:
                pct = (frame_index / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_index}/{total_frames} ({pct:.1f}%) — {len(events)} events this frame")

        exporter.flush()

    print(f"\nDone! Events written to: {output_path}")
    if output_video_path:
        print(f"Annotated video written to: {output_video_path}")


def _get_video_dimensions(video_path: str):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


class _NullContext:
    """No-op context manager used when video output is disabled."""
    def __enter__(self): return None
    def __exit__(self, *args): pass


def main():
    args = parse_args()
    run(
        video_path=args.video,
        config_path=args.config,
        output_path=args.output,
        output_video_path=args.output_video,
    )


if __name__ == "__main__":
    main()
