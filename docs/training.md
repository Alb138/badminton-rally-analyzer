# Training the Phase 2 Trackers

Two models are trained for the tracking comparison:
- the TrackNet-style heatmap model (`scripts/train_tracknet.py`)
- a YOLOv8 shuttle detector (`scripts/train_yolo.py`)

Both scripts run the same way locally and on Kaggle; only the paths differ.

## 1. Build the training data (local)

```bash
uv run python scripts/build_frame_cache.py --nycu-root data/raw/nycu --out data/cache/nycu_512x288
uv run python scripts/build_yolo_dataset.py --nycu-root data/raw/nycu --splits configs/splits/nycu.json \
    --out data/cache/yolo_nycu --stride 3
```

## 2a. Train on Kaggle (recommended)

Kaggle gives about 30 free GPU hours a week, and runs continue with the browser closed.

**One-time setup:**
1. On kaggle.com, go to **Settings → API → Create New Token** and save the downloaded `kaggle.json` as `~/.kaggle/kaggle.json`.
2. Phone-verify the account (Settings). Kaggle only allows GPU and internet in runs for verified accounts.

The Kaggle CLI runs through uv: `uvx --with six kaggle ...`.

**Upload the data as a private dataset.** It holds the NYCU data, which isn't ours to republish, so it must stay private.

```bash
uv run python scripts/kaggle_prepare.py --username YOUR_KAGGLE_NAME
uvx --with six kaggle datasets create -p data/cache --dir-mode zip      # private by default
```

**Run the training** (each script clones the `phase-2` branch of this repo on Kaggle):

```bash
uvx --with six kaggle kernels push -p outputs/kaggle_push/tracknet
uvx --with six kaggle kernels push -p outputs/kaggle_push/yolo
uvx --with six kaggle kernels status YOUR_KAGGLE_NAME/badminton-tracknet-train
```

Add `--smoke` to `kaggle_prepare.py` first to push 1-epoch versions (`badminton-*-train-smoke`). These check the Kaggle environment within minutes before you spend hours of quota.

**Download the weights when the status is `COMPLETE`:**

```bash
uvx --with six kaggle kernels output YOUR_KAGGLE_NAME/badminton-tracknet-train -p outputs/kaggle/tracknet
uvx --with six kaggle kernels output YOUR_KAGGLE_NAME/badminton-yolo-train -p outputs/kaggle/yolo
cp outputs/kaggle/tracknet/tracknet/best.pt models/tracknet_best.pt
cp outputs/kaggle/yolo/yolo/yolov8s/best.pt models/yolo_best.pt
```

Per-epoch metrics are in `tracknet/metrics.csv` and `yolo/yolov8s/results.csv` inside each output.

## 2b. Train locally

The TrackNet-style model fits a 4 GB GPU at batch size 2 to 4 and takes roughly overnight for 30 epochs:

```bash
uv run python scripts/train_tracknet.py --cache-root data/cache/nycu_512x288 \
    --splits-path configs/splits/nycu.json --out-dir outputs/runs/tracknet --batch-size 4
cp outputs/runs/tracknet/best.pt models/tracknet_best.pt
```

YOLOv8 at `imgsz=1280` is very slow on a 4 GB GPU (more than a day for 30 epochs), so train it on Kaggle.

## 3. Evaluate

```bash
uv run python scripts/eval_trackers.py --trackers tracknetv3 tracknet yolo --splits test amateur_test
```

Results go to `docs/results/tracking.md`.
