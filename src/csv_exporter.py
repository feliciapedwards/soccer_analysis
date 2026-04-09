import csv
from pathlib import Path
from typing import List

from .event_detector import Event

CSV_COLUMNS = [
    "frame_number",
    "timestamp_sec",
    "player_id",
    "team_id",
    "event_type",
    "player_x",
    "player_y",
    "ball_x",
    "ball_y",
    "ball_speed",
    "event_confidence",
]


class CSVExporter:
    """Streams events to a CSV file row-by-row (memory efficient for long games)."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

    def write_events(self, events: List[Event]):
        for event in events:
            self._writer.writerow({
                "frame_number": event.frame_number,
                "timestamp_sec": round(event.timestamp_sec, 4),
                "player_id": event.player_id,
                "team_id": event.team_id,
                "event_type": event.event_type,
                "player_x": _fmt(event.player_x),
                "player_y": _fmt(event.player_y),
                "ball_x": _fmt(event.ball_x),
                "ball_y": _fmt(event.ball_y),
                "ball_speed": round(event.ball_speed, 4),
                "event_confidence": round(event.event_confidence, 4),
            })

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _fmt(val) -> str:
    if val is None:
        return ""
    return str(round(val, 2))
