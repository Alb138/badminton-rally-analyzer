# Badminton Rally Analyzer

Upload a singles rally clip and get back an annotated video showing the shuttle's path, the players' skeletons, a label on each shot and a top-down court view, plus rally stats like rally length, shot counts and distance covered. Behind the demo are two model comparisons: a single-frame detector against a multi-frame tracker for the shuttle, and a pose-based model against a video-based model for recognizing shots.

**Status:** Phase 1 done. A rough end-to-end pipeline runs on pretrained parts (TrackNetV3 for the shuttle, YOLOv8-pose for players), and shot types show as `unknown` for now. Next come the tracking comparison (Phase 2) and shot recognition (Phase 3). Design: [`docs/specs/2026-11-15-badminton-analyzer-design.md`](docs/specs/2026-11-15-badminton-analyzer-design.md).

## Pipeline

```
rally.mp4
  ├─ shuttle tracker ────────► shuttle (x, y, visible) per frame
  ├─ YOLOv8-pose ────────────► keypoints for the 2 on-court players (filtered by court area)
  ├─ court homography ───────► player feet + hit points in court coordinates (metres)
  │   (4 clicked corners)
  ├─ hit detection ──────────► hit frames (shuttle direction change next to a player)
  ├─ shot classifier ────────► shot type per hit
  ├─ stats ──────────────────► rally summary (JSON)
  └─ renderer ───────────────► annotated.mp4
```

Every stage reads and writes two tables defined in `src/badminton_analyzer/schema.py`: a per-frame table and a per-shot table. That's why a stage can be swapped (for example, a new tracker in Phase 2) without touching the others.

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). PyTorch is installed with CUDA 12.8 wheels.

```bash
uv sync --extra dev --extra tracknetv3
```

**TrackNetV3** (pretrained shuttle tracker, Phase 1 baseline):

```bash
git clone https://github.com/qaz812345/TrackNetV3 third_party/TrackNetV3   # tested at commit 6eda442
# Download the checkpoints linked in its README and unzip them so that
# third_party/TrackNetV3/ckpts/TrackNet_best.pt and InpaintNet_best.pt exist.
```

The YOLOv8-pose weights download automatically to `third_party/weights/` on first use.

**Data** (not committed; see [`docs/data_audit.md`](docs/data_audit.md) for what's in it):
- NYCU Shuttlecock Trajectory Dataset: download `TrackNetV2.zip` from the [dataset page](https://hackmd.io/@TUIK/rJkRW54cU) (needs a browser) and unzip it into `data/raw/nycu/`.
- ShuttleSet: extract the `ShuttleSet/` folder of [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects) into `data/raw/shuttleset/`.

Then:

```bash
uv run python scripts/audit_data.py   # writes docs/data_audit.md and configs/splits/*.json
```

## Usage

The court needs four clicked corners (the outer doubles corners) on a frame that shows the main court view:

```bash
uv run python scripts/calibrate_court.py rally.mp4 --out cal.json --frame 100
uv run badminton-analyze rally.mp4 --calib cal.json --out outputs/rally
```

Outputs: `annotated.mp4`, `stats.json`, `frames.parquet`, `shots.parquet`.

Or use the local app (upload, pick a calibration frame, click the corners, Analyze):

```bash
uv run python -m badminton_analyzer.app
```

Input assumptions: one singles rally per clip, filmed from behind or above a baseline with all four outer court corners in view.

## Tests

```bash
uv run pytest
```

## Credits

- **NYCU Shuttlecock Trajectory Dataset**: per-frame shuttle labels used for tracking ([dataset page](https://hackmd.io/@TUIK/rJkRW54cU)).
- **ShuttleSet** (Wang et al., KDD 2023): stroke-level annotations of professional singles matches ([paper](https://arxiv.org/abs/2306.04948), [code](https://github.com/wywyWang/CoachAI-Projects)).
- **TrackNetV3** (MIT): pretrained shuttle tracker and checkpoints, used as the Phase 1 baseline ([repo](https://github.com/qaz812345/TrackNetV3), commit `6eda442`).
- **Ultralytics YOLOv8**: pose estimation and detection (AGPL-3.0).
