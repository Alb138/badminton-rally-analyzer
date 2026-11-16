# Phase 2: Shuttle Tracking Comparison: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare a single-frame YOLOv8 shuttle detector against our own multi-frame TrackNet-style heatmap model on the NYCU splits, using TrackNet's published scoring rules. Report precision, recall, F1 and speed for broadcast (official test) and amateur footage, and swap the winner into the pipeline.

**Architecture:**
- Both new trackers implement the existing `ShuttleTracker` protocol (`tracking/base.py`), so the pipeline, app and evaluation treat them like TrackNetV3.
- A `make_tracker(cfg)` factory picks the tracker from `configs/pipeline.yaml`.
- One evaluation module scores any tracker against NYCU labels. Pretrained TrackNetV3 stays in the results as a reference row.
- Training code lives in the package and runs the same way locally or in a Kaggle notebook. Only paths differ, and they're passed as CLI arguments.

**Tech Stack:** PyTorch 2.11 (CUDA 12.8), Ultralytics 8.4 (YOLOv8), OpenCV, pandas, pytest; Kaggle notebooks for the long training runs.

**Spec:** `docs/specs/2026-11-15-badminton-analyzer-design.md` (Goals: Comparison 1; Phases: 2. Tracking comparison; Anti-goals).

## Global Constraints

- **Scoring rules** (exactly as in TrackNetV3's `test.py` and `utils/metric.py`). Each frame is one of:
  - **TP:** both visible, and distance ≤ tolerance
  - **FP1:** both visible, and distance > tolerance
  - **FP2:** predicted visible, label invisible
  - **FN:** predicted invisible, label visible
  - **TN:** both invisible
- **Formulas:**
  - precision = TP / (TP + FP1 + FP2)
  - recall = TP / (TP + FN)
  - F1 = 2PR / (P + R)
  - accuracy = (TP + TN) / all
  - Any zero denominator gives 0.
- **Tolerance:** 4 px in a 512-px-wide frame, scaled by the video's width: `4 * width / 512`, which is 10 px at 1280×720.
- **Splits:** train on `configs/splits/nycu.json` `train`, select checkpoints on `val`, and report only on `test` (official) and `amateur_test`. Never tune on test.
- **Seeds and configs:** seed `42`. Every training run writes its config and per-epoch metrics next to its checkpoint.
- **TrackNet-style model:** input 3 consecutive frames at 512×288, RGB, scaled to [0,1]; output 3 heatmaps (one per frame). Gaussian targets with σ = 2.5 px in input space.
- **YOLOv8:** full-resolution frames, `imgsz=1280`, single class `shuttle`, 16×16 px boxes centred on the labelled point (at the source resolution).
- **Weights and data never committed:** `*.pt` is already ignored. Caches go under `/data/` and run outputs under `/outputs/`.
- **Local GPU:** RTX 3050 Laptop with 4 GB. Inference batch sizes stay small, configurable, and default to fitting in 4 GB.
- **Kaggle:** code must not hard-code Kaggle paths; the notebook passes them in.
- **Attribution:** commits use the user's git identity only. No AI credit anywhere.

## Review Focus

1. **Clip lengths that aren't a multiple of 3, or shorter than 3 frames.** The TrackNet-style tracker must still return exactly one row per decoded frame (pad by repeating the last frame). Test: Task 5.
2. **Label rows and decoded frames disagree in count** (NYCU CSVs vs. video decoding). Scoring aligns on `frame`: missing predictions count as invisible, and predictions without a label row are ignored. Test: Task 1.
3. **Labels marked visible with coordinates outside the frame, or invisible rows carrying `0,0`.** No heatmap is drawn for invisible frames, and out-of-frame points are clipped. Test: Task 2.
4. **Sources that aren't 1280×720** (1080p, 60 fps phone clips). Both new trackers scale coordinates back to the source size, and the tolerance scales with width. Tests: Tasks 1, 5 and 7.
5. **A match appearing in two splits.** Training must refuse to start. Test: Task 4.

---

### Task 1: Scoring module, tracker factory and the TrackNetV3 baseline numbers

**Files:**
- Create: `src/badminton_analyzer/tracking/evaluate.py`, `src/badminton_analyzer/tracking/factory.py`, `scripts/eval_trackers.py`, `tests/test_evaluate.py`, `tests/test_factory.py`
- Modify: `src/badminton_analyzer/pipeline.py` (`build_default_components` uses `make_tracker`)
- Generated: `docs/results/tracking.md`

**Interfaces:**
- **Consumes:** `NycuRally`, `index_nycu`, `read_ball_csv` (`data/nycu.py`); `load_splits` (`data/splits.py`); `read_video_meta`; `TrackNetV3Tracker`.
- **Produces, `evaluate.py`:**
  - `TOLERANCE_AT_512 = 4.0`
  - `tolerance_px(frame_width: int) -> float`
  - `@dataclass Counts(tp=0, tn=0, fp1=0, fp2=0, fn=0)` with `__add__`
  - `count_outcomes(gt: pd.DataFrame, pred: pd.DataFrame, tolerance: float) -> Counts`. Both tables have `frame, visible, x, y`; rows are aligned on `frame` from `gt`.
  - `scores(c: Counts) -> dict[str, float]` with keys `precision, recall, f1, accuracy`
  - `@dataclass RallyResult(match_id: str, rally_id: str, counts: Counts, n_frames: int, seconds: float)`
  - `evaluate_tracker(tracker, rallies: list[NycuRally]) -> list[RallyResult]`: times each `tracker.track()` call.
  - `summarize(results: list[RallyResult]) -> dict` with the summed counts, `scores(...)`, `frames`, `seconds`, and `fps = frames / seconds`.
- **Produces, `factory.py`:** `make_tracker(cfg: dict) -> ShuttleTracker`. Kind `"tracknetv3"` is the existing construction moved out of `pipeline.py`. Unknown kinds raise `ValueError` naming the kind. Tasks 5 and 7 add kinds `"tracknet"` and `"yolo"`.
- **Produces, `eval_trackers.py`:**
  - CLI: `--trackers NAME [NAME ...]` (each NAME is a section of `configs/trackers.yaml` passed to `make_tracker`), `--splits test amateur_test`, `--nycu-root data/raw/nycu`, `--splits-file configs/splits/nycu.json`, `--out docs/results/tracking.md`, `--force`.
  - Each (tracker, split) summary is cached in `outputs/eval/<tracker>_<split>.json` and reused unless `--force`. The markdown is rebuilt from every cached summary, so earlier rows (e.g. TrackNetV3) aren't recomputed.
  - The markdown has one table per split: tracker, precision, recall, F1, accuracy, TP, FP1, FP2, FN, fps.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
def table(rows):  # rows: (frame, visible, x, y)
    return pd.DataFrame(rows, columns=["frame", "visible", "x", "y"])

def test_count_outcomes_follows_tracknet_rules():
    gt = table([(0, True, 100, 100), (1, True, 100, 100), (2, False, np.nan, np.nan), (3, True, 50, 50), (4, False, np.nan, np.nan)])
    pred = table([(0, True, 105, 105), (1, True, 120, 100), (2, True, 10, 10), (3, False, np.nan, np.nan), (4, False, np.nan, np.nan)])
    assert count_outcomes(gt, pred, tolerance=10.0) == Counts(tp=1, fp1=1, fp2=1, fn=1, tn=1)

def test_missing_predictions_count_as_invisible_and_extra_rows_are_ignored():
    gt = table([(0, True, 10, 10), (1, True, 10, 10)])
    pred = table([(0, True, 10, 10), (5, True, 1, 1)])
    assert count_outcomes(gt, pred, 10.0) == Counts(tp=1, fn=1)

def test_scores_use_tracknet_formulas():
    s = scores(Counts(tp=8, tn=5, fp1=1, fp2=1, fn=2))
    assert s["precision"] == pytest.approx(0.8) and s["recall"] == pytest.approx(0.8)
    assert s["f1"] == pytest.approx(0.8) and s["accuracy"] == pytest.approx(13 / 17)
    assert scores(Counts())["f1"] == 0.0

def test_tolerance_scales_with_width():
    assert tolerance_px(512) == 4.0 and tolerance_px(1280) == 10.0 and tolerance_px(1920) == 15.0

def test_evaluate_tracker_scores_a_perfect_tracker(tmp_path):
    video = make_video(tmp_path / "v.mp4", n_frames=6)
    labels = tmp_path / "1_01_00_ball.csv"
    labels.write_text("Frame,Visibility,X,Y\n" + "".join(
        f"{i},{0 if i == 2 else 1},{0 if i == 2 else 40},{0 if i == 2 else 20}\n" for i in range(6)))
    rally = NycuRally("Test/match1", "1_01_00", video, labels)

    class Perfect:
        def track(self, video_path):
            return read_ball_csv(labels)

    [result] = evaluate_tracker(Perfect(), [rally])
    assert result.counts == Counts(tp=5, tn=1) and result.n_frames == 6 and result.seconds >= 0
```

```python
# tests/test_factory.py
def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        make_tracker({"kind": "nope"})

def test_tracknetv3_kind_builds_wrapper(tmp_path):
    t = make_tracker({"kind": "tracknetv3", "repo_dir": str(tmp_path), "tracknet_ckpt": "a.pt",
                      "inpaintnet_ckpt": "b.pt", "work_dir": str(tmp_path / "w"), "batch_size": 2})
    assert isinstance(t, TrackNetV3Tracker) and t.batch_size == 2
```

- [ ] **Step 2: Run the tests.** Expected: FAIL (`ModuleNotFoundError`). Run: `uv run pytest tests/test_evaluate.py tests/test_factory.py -v`
- [ ] **Step 3: Implement.**
  - Write `evaluate.py` and `factory.py`.
  - Make `pipeline.build_default_components` call `make_tracker(cfg.tracker)`.
  - Add `configs/trackers.yaml` with a `tracknetv3:` section, copied from `pipeline.yaml`'s tracker block.
  - Write `scripts/eval_trackers.py`.
- [ ] **Step 4: Run the full suite.** Expected: all PASS. Run: `uv run pytest -q`
- [ ] **Step 5: Baseline run** (GPU, about 1.5 h; run it in the background).

  Run: `uv run python scripts/eval_trackers.py --trackers tracknetv3 --splits test amateur_test --out docs/results/tracking.md`

  Expected: `docs/results/tracking.md` has one TrackNetV3 row per split, and broadcast F1 is in the published range (above 0.9). Under the table, note that TrackNetV3's training data likely included the amateur matches.
- [ ] **Step 6: Commit** with `git commit -m "feat: TrackNet-protocol tracker evaluation, tracker factory and TrackNetV3 baseline"`

---

### Task 2: Frame cache and training windows for the TrackNet-style model

**Files:**
- Create: `src/badminton_analyzer/tracking/frames_cache.py`, `src/badminton_analyzer/tracking/tracknet_data.py`, `scripts/build_frame_cache.py`, `tests/test_tracknet_data.py`

**Interfaces:**
- **Produces, `frames_cache.py`:**
  - `INPUT_SIZE = (512, 288)` (w, h)
  - `cache_rally(video_path: Path, label_path: Path, out_dir: Path, size=INPUT_SIZE, quality=95) -> int`:
    - writes `out_dir/00000.jpg ...` (resized), `labels.csv` (a copy of the source labels) and `meta.json` `{"width", "height", "fps", "n_frames"}` (source values)
    - returns the frame count
    - skips the rally if `meta.json` exists with the same `n_frames`
- **CLI:** `scripts/build_frame_cache.py --nycu-root data/raw/nycu --out data/cache/nycu_512x288` caches every rally under `<out>/<match_id>/<rally_id>/`.
- **Produces, `tracknet_data.py`:**
  - `SEQ_LEN = 3`, `SIGMA = 2.5`
  - `gaussian_heatmap(x: float, y: float, visible: bool, size=(512, 288), sigma=SIGMA) -> np.ndarray` (H, W) float32 with peak 1.0. Returns all zeros if not visible. The point is clipped to the frame.
  - `class RallyWindows(torch.utils.data.Dataset)`:
    - `__init__(rally_dirs: list[Path], size=(512, 288), stride=1, augment=False, flip_p=0.5, seed=42)`
    - one item per window start `0, stride, ...` with `start + SEQ_LEN <= n_frames`
    - `__getitem__ -> (x: float32 (3*SEQ_LEN, H, W) in [0,1], y: float32 (SEQ_LEN, H, W), coords: float32 (SEQ_LEN, 3) as [visible, x, y] in input pixels)`
    - labels are scaled from the source size using `meta.json`
    - `augment=True` flips the whole window horizontally with probability `flip_p`, mirroring x as `W - 1 - x`; the decision comes from `np.random.default_rng(seed + index)`

- [ ] **Step 1: Write the failing tests**

```python
def test_heatmap_peaks_at_the_point():
    hm = gaussian_heatmap(100.0, 50.0, True)
    assert hm.shape == (288, 512) and hm.max() == pytest.approx(1.0)
    assert np.unravel_index(hm.argmax(), hm.shape) == (50, 100)

def test_invisible_heatmap_is_empty_even_with_zero_coords():
    assert not gaussian_heatmap(0.0, 0.0, False).any()

def test_out_of_frame_point_is_clipped():
    hm = gaussian_heatmap(600.0, -5.0, True)
    assert np.unravel_index(hm.argmax(), hm.shape) == (0, 511)

def rally_cache(tmp_path):  # 7 frames at 128x64; shuttle at source (40, 20), invisible on frame 3
    video = make_video(tmp_path / "v.mp4", n_frames=7, size=(128, 64))
    labels = tmp_path / "1_01_00_ball.csv"
    labels.write_text("Frame,Visibility,X,Y\n" + "".join(
        f"{i},{0 if i == 3 else 1},{0 if i == 3 else 40},{0 if i == 3 else 20}\n" for i in range(7)))
    return cache_rally(video, labels, tmp_path / "c", size=(64, 32)), tmp_path / "c"

def test_cache_and_windows(tmp_path):
    n, cache = rally_cache(tmp_path)
    ds = RallyWindows([cache], size=(64, 32))
    x, y, coords = ds[0]
    assert n == 7 and len(ds) == 5 and x.shape == (9, 32, 64) and y.shape == (3, 32, 64)
    assert coords[0].tolist() == [1.0, 20.0, 10.0]   # (40, 20) scaled by 64/128
    assert not ds[1][1][2].any()                     # window 1 = frames 1-3; frame 3 invisible -> empty map

def test_flip_mirrors_image_and_label(tmp_path):
    _, cache = rally_cache(tmp_path)
    _, y, coords = RallyWindows([cache], size=(64, 32), augment=True, flip_p=1.0)[0]
    assert coords[0, 1] == pytest.approx(63 - 20.0)
    assert np.unravel_index(int(y[0].argmax()), y[0].shape)[1] == 43
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_tracknet_data.py -v`
- [ ] **Step 3: Implement `frames_cache.py`, `tracknet_data.py` and `scripts/build_frame_cache.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Build the real cache.**

  Run: `uv run python scripts/build_frame_cache.py --nycu-root data/raw/nycu --out data/cache/nycu_512x288`

  Expected: 201 rally folders, about 91k JPEGs, about 2 GB.
- [ ] **Step 6: Commit** with `git commit -m "feat: frame cache and 3-frame training windows for the TrackNet-style model"`

---

### Task 3: TrackNet-style model, loss and heatmap decoding

**Files:**
- Create: `src/badminton_analyzer/tracking/tracknet_model.py`, `tests/test_tracknet_model.py`

**Interfaces:**
- `class TrackNetV2(nn.Module)`: `__init__(in_frames: int = 3, out_frames: int = 3)`.
  - **Encoder:** VGG-style blocks of 64-64 | pool | 128-128 | pool | 256-256-256 | pool | 512-512-512, each conv being Conv3×3 + BatchNorm + ReLU.
  - **Decoder:** upsample ×2 + skip concat at each level, then 256-256-256 | 128-128 | 64-64.
  - **Output:** a 1×1 conv to `out_frames` channels, then sigmoid.
  - **Forward:** `(B, 3*in_frames, H, W) -> (B, out_frames, H, W)`. H and W must be divisible by 8.
- `wbce_loss(pred: Tensor, target: Tensor) -> Tensor` is TrackNetV2's weighted BCE: `-mean((1-p)^2 * y * log(p) + p^2 * (1-y) * log(1-p))`, with p clamped to [1e-7, 1-1e-7].
- `decode_heatmap(hm: np.ndarray, threshold: float = 0.5) -> tuple[bool, float, float]` binarizes at the threshold and returns the centroid of the largest connected component (`cv2.connectedComponentsWithStats`), or `(False, nan, nan)` if nothing is above the threshold.

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_shape_and_range():
    m = TrackNetV2().eval()
    out = m(torch.rand(2, 9, 32, 64))
    assert out.shape == (2, 3, 32, 64) and 0 <= out.min() and out.max() <= 1

def test_parameter_count_is_tracknetv2_sized():
    n = sum(p.numel() for p in TrackNetV2().parameters())
    assert 5e6 < n < 20e6

def test_loss_prefers_the_right_heatmap():
    target = torch.zeros(1, 3, 32, 64); target[0, :, 10, 20] = 1.0
    good = target.clamp(0.01, 0.99); bad = (1 - target).clamp(0.01, 0.99)
    assert wbce_loss(good, target) < wbce_loss(bad, target) and torch.isfinite(wbce_loss(good, target))

def test_decode_returns_centroid_of_largest_blob():
    hm = np.zeros((32, 64), np.float32); hm[10:13, 20:23] = 0.9; hm[1, 1] = 0.9
    assert decode_heatmap(hm) == (True, pytest.approx(21.0), pytest.approx(11.0))

def test_decode_empty_heatmap_is_invisible():
    visible, x, y = decode_heatmap(np.full((32, 64), 0.2, np.float32))
    assert not visible and np.isnan(x) and np.isnan(y)
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_tracknet_model.py -v`
- [ ] **Step 3: Implement `tracknet_model.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: TrackNet-style heatmap model, weighted BCE loss and decoding"`

---

### Task 4: Training loop

**Files:**
- Create: `src/badminton_analyzer/tracking/train_tracknet.py`, `scripts/train_tracknet.py`, `tests/test_train_tracknet.py`

**Interfaces:**
- **Consumes:** `RallyWindows`, `TrackNetV2`, `wbce_loss`, `decode_heatmap` (Tasks 2–3); `count_outcomes`, `scores` (Task 1); `load_splits`.
- **Produces:**
  - `@dataclass TrainConfig(cache_root: Path, splits_path: Path, out_dir: Path, epochs: int = 30, batch_size: int = 8, lr: float = 1e-3, num_workers: int = 2, seed: int = 42, amp: bool = True, size: tuple[int, int] = (512, 288), val_rallies: int | None = None, device: str | None = None)`
  - `train(cfg: TrainConfig) -> Path` returns the best checkpoint path, `out_dir/best.pt`, chosen by val F1.
- **Data:**
  - Training windows use `stride=1` with augmentation on.
  - Validation decodes `stride=3` windows on the `val` matches, optionally only the first `val_rallies` rallies, and scores them with `count_outcomes` at a tolerance of 4 px in input space.
- **Training settings:** AdamW, lr decayed 10× at 2/3 of the epochs, and seeds set for `random`, `numpy` and `torch`.
- **Files written:**
  - `out_dir/config.json`
  - `out_dir/metrics.csv` with columns `epoch, train_loss, val_precision, val_recall, val_f1, seconds`
  - `best.pt` and `last.pt` containing `{"model": state_dict, "epoch", "val_f1", "size"}`
- **Safety:** raises `ValueError` naming the match if any id is in more than one of train, val and test.
- **CLI:** `scripts/train_tracknet.py` exposes every `TrainConfig` field as a flag.

- [ ] **Step 1: Write the failing tests** (tiny synthetic cache: 3 matches × 1 rally × 9 frames at 64×32, built with `cache_rally`, plus a splits JSON written by the test)

```python
def test_one_epoch_writes_checkpoint_and_metrics(tmp_path):
    best = train(TrainConfig(cache, splits, tmp_path / "run", epochs=1, batch_size=2, num_workers=0,
                             size=(64, 32), amp=False, device="cpu"))
    assert best.exists() and torch.load(best)["size"] == [64, 32]
    assert len(pd.read_csv(tmp_path / "run" / "metrics.csv")) == 1
    assert json.loads((tmp_path / "run" / "config.json").read_text())["seed"] == 42

def test_overlapping_splits_are_rejected(tmp_path):
    # same match id in train and test
    with pytest.raises(ValueError, match="Professional/m1"):
        train(TrainConfig(cache, leaky_splits, tmp_path / "run", epochs=1, num_workers=0, size=(64, 32), device="cpu"))
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_train_tracknet.py -v`
- [ ] **Step 3: Implement `train_tracknet.py` and the CLI.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Local smoke run on the real cache** (GPU, about 2 min).

  Run: `uv run python scripts/train_tracknet.py --cache-root data/cache/nycu_512x288 --splits-path configs/splits/nycu.json --out-dir outputs/runs/tracknet_smoke --epochs 1 --batch-size 2 --val-rallies 2`

  Expected: `metrics.csv` has 1 row, the loss is finite, and peak GPU memory stays under 4 GB. If it doesn't fit, record the batch size that does in the config notes.
- [ ] **Step 6: Commit** with `git commit -m "feat: TrackNet-style training loop with val-F1 checkpointing"`

---

### Task 5: Own TrackNet-style tracker for inference

**Files:**
- Create: `src/badminton_analyzer/tracking/tracknet_tracker.py`, `tests/test_tracknet_tracker.py`
- Modify: `src/badminton_analyzer/tracking/factory.py` (kind `"tracknet"`), `configs/trackers.yaml`

**Interfaces:**
- `TrackNetTracker(weights: Path | None, device: str | None = None, batch_size: int = 4, threshold: float = 0.5, model: nn.Module | None = None, size: tuple[int, int] | None = None)`:
  - With real weights, the model and `size` come from the checkpoint. `model` + `size` let tests inject a stub without a checkpoint.
  - `track(video_path) -> DataFrame[frame, visible, x, y]`: decodes and resizes the frames, then runs non-overlapping 3-frame windows, padding the last window by repeating the final frame.
  - Each frame's heatmap is decoded with `decode_heatmap`, and its coordinates are scaled back to the source size.
  - Returns exactly one row per decoded frame.
- **Factory:** `make_tracker({"kind": "tracknet", "weights": ..., "batch_size": ..., "threshold": ...})` raises `FileNotFoundError` naming the weights path if it's missing.

- [ ] **Step 1: Write the failing tests**

```python
class BlobModel(nn.Module):  # outputs a blob at input (x=32, y=16) for every frame
    def forward(self, x):
        out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3]); out[:, :, 15:18, 31:34] = 0.9
        return out

def test_one_row_per_frame_for_lengths_not_divisible_by_three(tmp_path):
    for n in (1, 2, 7):
        df = TrackNetTracker(None, model=BlobModel(), size=(64, 32)).track(make_video(tmp_path / f"{n}.mp4", n_frames=n, size=(128, 64)))
        assert df["frame"].tolist() == list(range(n))

def test_coordinates_are_scaled_to_source_size(tmp_path):
    df = TrackNetTracker(None, model=BlobModel(), size=(64, 32)).track(make_video(tmp_path / "v.mp4", n_frames=3, size=(128, 64)))
    assert df["visible"].all() and df["x"].iloc[0] == pytest.approx(64.0) and df["y"].iloc[0] == pytest.approx(32.0)

def test_factory_requires_existing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        make_tracker({"kind": "tracknet", "weights": str(tmp_path / "missing.pt")})
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_tracknet_tracker.py -v`
- [ ] **Step 3: Implement `tracknet_tracker.py`** and the factory kind.
- [ ] **Step 4: Run the full suite.** Expected: all PASS. Run: `uv run pytest -q`
- [ ] **Step 5: Commit** with `git commit -m "feat: inference tracker for the TrackNet-style model"`

---

### Task 6: YOLO dataset from point labels

**Files:**
- Create: `src/badminton_analyzer/tracking/yolo_data.py`, `scripts/build_yolo_dataset.py`, `tests/test_yolo_data.py`

**Interfaces:**
- `BOX_PX = 16`
- `yolo_label_line(x: float, y: float, width: int, height: int, box_px: int = BOX_PX) -> str` returns `"0 cx cy w h"`, normalized and to 6 decimals, with the box clipped to the image.
- `build_yolo_dataset(rallies: dict[str, list[NycuRally]], out_root: Path, stride: int = 3, quality: int = 90) -> Path`:
  - For each split (`train`, `val`), writes `images/<split>/<match>__<rally>__<frame:05d>.jpg` (full resolution, every `stride`-th frame) and the matching `labels/<split>/*.txt`. The `/` in match ids becomes `-`.
  - Invisible frames get an empty label file and are kept as negatives.
  - Writes `data.yaml` (`path`, `train: images/train`, `val: images/val`, `names: {0: shuttle}`) and returns its path.
- **CLI:** `scripts/build_yolo_dataset.py --nycu-root data/raw/nycu --splits configs/splits/nycu.json --out data/cache/yolo_nycu --stride 3`

- [ ] **Step 1: Write the failing tests**

```python
def test_label_line_is_normalized():
    assert yolo_label_line(640, 360, 1280, 720) == "0 0.500000 0.500000 0.012500 0.022222"

def test_label_box_is_clipped_at_the_border():
    cx, cy, w, h = map(float, yolo_label_line(2, 2, 1280, 720).split()[1:])
    assert cx - w / 2 >= 0 and cy - h / 2 >= 0

def test_build_dataset_strides_and_keeps_negatives(tmp_path):
    # one 7-frame 128x64 rally in train, shuttle at (40, 20), invisible on frame 3 (same CSV as Task 2's fixture)
    rally = NycuRally("Professional/match1", "1_01_00", video, labels)
    data_yaml = build_yolo_dataset({"train": [rally], "val": []}, tmp_path / "y", stride=3)
    imgs = sorted((tmp_path / "y" / "images" / "train").glob("*.jpg"))
    assert [p.stem.split("__")[-1] for p in imgs] == ["00000", "00003", "00006"]
    assert (tmp_path / "y" / "labels" / "train" / (imgs[1].stem + ".txt")).read_text() == ""
    assert "names" in data_yaml.read_text()
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_yolo_data.py -v`
- [ ] **Step 3: Implement `yolo_data.py` and the CLI.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Build the real dataset.**

  Run: `uv run python scripts/build_yolo_dataset.py --nycu-root data/raw/nycu --splits configs/splits/nycu.json --out data/cache/yolo_nycu --stride 3`

  Expected: about 19k train and about 3k val images, about 3 GB.
- [ ] **Step 6: Commit** with `git commit -m "feat: YOLO shuttle dataset from NYCU point labels"`

---

### Task 7: YOLO training script and YOLO tracker

**Files:**
- Create: `scripts/train_yolo.py`, `src/badminton_analyzer/tracking/yolo_tracker.py`, `tests/test_yolo_tracker.py`
- Modify: `src/badminton_analyzer/tracking/factory.py` (kind `"yolo"`), `configs/trackers.yaml`

**Interfaces:**
- **`scripts/train_yolo.py`:**
  - Flags: `--data`, `--model yolov8s.pt`, `--imgsz 1280`, `--epochs 30`, `--batch -1`, `--fraction 1.0`, `--out outputs/runs/yolo`, `--name`.
  - Calls `YOLO(model).train(data=..., imgsz=..., epochs=..., batch=..., fraction=..., project=out, name=name, seed=42, deterministic=True, single_cls=True)`.
  - Copies `weights/best.pt` to `<out>/<name>/best.pt` and prints its path.
- **`YoloShuttleTracker(weights: Path | None, conf: float = 0.25, imgsz: int = 1280, device: str | None = None, predict_fn: Callable[[np.ndarray], np.ndarray] | None = None)`:**
  - `predict_fn(frame) -> (N, 3)` array of `[cx, cy, conf]` in source pixels. The default wraps `YOLO(weights).predict(frame, imgsz=..., conf=..., verbose=False)`.
  - `track` keeps the highest-confidence box per frame; frames with no box are invisible.
- **Factory:** `make_tracker({"kind": "yolo", "weights", "conf", "imgsz"})`, with the same missing-weights error as Task 5.

- [ ] **Step 1: Write the failing tests**

```python
def test_highest_confidence_box_wins(tmp_path):
    boxes = {0: np.array([[10, 10, 0.3], [50, 20, 0.9]]), 1: np.zeros((0, 3))}
    calls = iter(range(2))
    t = YoloShuttleTracker(None, predict_fn=lambda frame: boxes[next(calls)])
    df = t.track(make_video(tmp_path / "v.mp4", n_frames=2))
    assert df["visible"].tolist() == [True, False] and (df.loc[0, "x"], df.loc[0, "y"]) == (50, 20)

def test_factory_requires_existing_yolo_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        make_tracker({"kind": "yolo", "weights": str(tmp_path / "nope.pt")})
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_yolo_tracker.py -v`
- [ ] **Step 3: Implement `yolo_tracker.py`, the factory kind and `scripts/train_yolo.py`.**
- [ ] **Step 4: Run the full suite.** Expected: all PASS.
- [ ] **Step 5: Local smoke run** (GPU, 1 epoch on about 1% of the images): `uv run python scripts/train_yolo.py --data data/cache/yolo_nycu/data.yaml --epochs 1 --imgsz 1280 --batch 2 --fraction 0.01 --name smoke`. Expected: it finishes and prints the `best.pt` path.
- [ ] **Step 6: Commit** with `git commit -m "feat: YOLOv8 shuttle detector training and tracker"`

---

### Task 8: Kaggle training notebook and the training checkpoint

**Files:**
- Create: `notebooks/kaggle_train.ipynb`, `docs/training.md`

**Notebook cells:**
1. `pip install git+https://github.com/Alb138/badminton-rally-analyzer@phase-2`
2. Paths from the attached Kaggle datasets: `/kaggle/input/<name>/nycu_512x288`, `/kaggle/input/<name>/yolo_nycu`, and the splits file shipped with the repo.
3. TrackNet-style training via `scripts/train_tracknet.py` with `--batch-size 8 --epochs 30`.
4. YOLO training via `scripts/train_yolo.py` with `--epochs 30 --imgsz 1280`.
5. Copy `best.pt` and both `metrics.csv` / `results.csv` to `/kaggle/working/` for download.

**`docs/training.md`:**
- how to zip and upload the two caches as one **private** Kaggle dataset (NYCU data isn't ours to republish)
- attaching it, choosing GPU, and running with "Save Version → Save & Run All" so it continues without the browser open
- downloading the weights into `models/`
- the local alternative for the TrackNet-style run (overnight on the 4 GB GPU with `--batch-size 2`)

- [ ] **Step 1: Write the notebook and `docs/training.md`.** Validate the notebook JSON with `uv run python -c "import json; json.load(open('notebooks/kaggle_train.ipynb'))"`.
- [ ] **Step 2: Commit and push** (the notebook installs the package from this branch): `git commit -m "docs: Kaggle training notebook and guide"`, then `git push -u origin phase-2`
- [ ] **Step 3: CHECKPOINT. Stop and hand over to the user.** The user uploads the dataset and runs the notebook, or chooses local training. Resume at Task 9 once `models/tracknet_best.pt` and `models/yolo_best.pt` exist.

---

### Task 9: Final comparison, swap in the winner, document

**Files:**
- Modify: `configs/trackers.yaml` (weights paths), `configs/pipeline.yaml` (`tracker:` = winner), `docs/results/tracking.md`, `README.md`

- [ ] **Step 1: Evaluate both new trackers on test and amateur.** The TrackNetV3 rows are already cached from Task 1.

  Run: `uv run python scripts/eval_trackers.py --trackers tracknetv3 tracknet yolo --splits test amateur_test --out docs/results/tracking.md`

  Expected: three rows per split with P/R/F1/accuracy, outcome counts and fps.
- [ ] **Step 2: Write the analysis under the tables in `docs/results/tracking.md`:**
  - whether the target F1 ≥ 0.85 on broadcast was met
  - where each model fails (FP1 vs FP2 vs FN)
  - the speed trade-off
  - the amateur gap
  - caveats: TrackNetV3 may have seen the amateur matches; the test set is only 3 matches
- [ ] **Step 3: Pick the winner.** It's the higher broadcast-test F1; if within 0.01, the faster one. Set it as `configs/pipeline.yaml`'s `tracker:` section.
- [ ] **Step 4: Rerun the pipeline** on `Test/match1/1_05_03` with the new tracker and check the video by eye.

  Run: `uv run badminton-analyze data/raw/nycu/TrackNetV2/TrackNetV2/Test/match1/video/1_05_03.mp4 --calib configs/calibrations/nycu_test_match1_1_05_03.json --out outputs/phase2_check`
- [ ] **Step 5: Update the README:** status Phase 2 done, the results table, and how to download or train weights.
- [ ] **Step 6: Full suite and commit.** Run `uv run pytest -q`, then `git commit -m "docs: Phase 2 tracking comparison results; switch pipeline to <winner>"`
