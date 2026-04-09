# Soccer Analysis — Event Dataset Extractor

Extracts a structured event dataset (CSV) from iPhone soccer game recordings using YOLOv8 + ByteTrack.

## Detected Events

| Event | Description |
|---|---|
| `BALL_POSITION` | Ball location every frame |
| `POSSESSION` | Player with ball closest to them |
| `PASS` | Possession transfer between players |
| `SHOT` | High-speed ball movement away from player |
| `GOAL` | Ball enters a defined goal region |
| `FOUL` | Player collision / sudden velocity drop |
| `OUT_OF_BOUNDS` | Ball exits defined pitch boundary |

## Output Schema (`output/events.csv`)

| Column | Description |
|---|---|
| `frame_number` | Integer frame index |
| `timestamp_sec` | Seconds from video start |
| `player_id` | ByteTrack integer ID (reset per video) |
| `team_id` | 0 or 1 (jersey color cluster) |
| `event_type` | Event label |
| `player_x` | Center x of player bounding box (pixels) |
| `player_y` | Center y of player bounding box (pixels) |
| `ball_x` | Center x of ball (pixels) |
| `ball_y` | Center y of ball (pixels) |
| `ball_speed` | Pixels/frame ball displacement |
| `event_confidence` | Heuristic confidence score 0–1 |

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python src/main.py --video data/game.mp4 --config config.yaml --output output/events.csv
```

## Run on Google Colab

Open `notebooks/soccer_event_extraction.ipynb` and follow the cells. Mount your Google Drive and point to your video file.

## Configuration

Edit `config.yaml` to set:
- Goal region pixel bounding boxes (per video)
- Pitch boundary bounding box
- Possession distance threshold
- Shot speed threshold
