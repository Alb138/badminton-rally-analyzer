# Phases 0–1: Setup, Data Audit and Rough End-to-End Pipeline: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the repo, audit both datasets (including whether the ShuttleSet videos can still be fetched), and ship a rough pipeline that turns a rally clip into an annotated video and a stats JSON, using pretrained parts.

**Architecture:**
- A `src/` Python package. Each pipeline stage (tracker, pose, court, hits, shots, stats, render) reads and writes two pandas tables defined in `schema.py`: a per-frame table and a per-shot table.
- Heavy models sit behind small Protocols (`ShuttleTracker`, `PoseEstimator`, `ShotClassifier`). Tests use fakes, and later phases swap in new models without touching the other stages.
- TrackNetV3 runs as a subprocess from a gitignored clone.

**Tech Stack:**
- Python 3.12 and uv
- PyTorch (CUDA 12.8 wheels)
- Ultralytics YOLOv8-pose
- OpenCV, pandas, pyarrow
- imageio-ffmpeg (H.264 output)
- Gradio, yt-dlp, pytest

**Spec:** `docs/specs/2026-11-15-badminton-analyzer-design.md`

## Global Constraints

- **Python and environment:** Python 3.12 (`requires-python = ">=3.12,<3.13"`), environment managed by uv, `uv.lock` committed.
- **PyTorch:** `torch` and `torchvision` come from the `https://download.pytorch.org/whl/cu128` index. Local GPU is an RTX 3050 Laptop with 4 GB.
- **Scope:** singles only. Input is a single rally clip in a broadcast-style view from behind or above the baseline.
- **Court coordinates (metres):**
  - origin at the far-left outer (doubles) corner as seen in the video
  - x runs to the right, 0–6.10
  - y runs toward the camera, 0–13.40
  - net at y = 6.70
- **Player ids:** exactly `"near"` and `"far"`.
- **Splits:** all train/val/test splits are by match, seed `42`, and the split files are committed.
- **Never committed:** `data/`, `third_party/`, `outputs/`, model weights, dataset videos or frames.
- **Attribution:**
  - Commits use the user's existing git identity only. No `Co-Authored-By` trailers, and no AI tool or assistant credited anywhere (README, `pyproject.toml` authors, CONTRIBUTORS).
  - Third-party parts (NYCU dataset, ShuttleSet, TrackNetV3, Ultralytics) are credited in the README.
- **Paths:** the repo lives under a path with spaces (`.../Albert Mario Laurent/Projects/Project 2`). Use `pathlib.Path` everywhere. Subprocesses get list arguments, never a shell string.

## Review Focus

1. **Paths containing spaces** (the repo's own location, user clip names like `rally 1.mp4`). Every stage, including the TrackNetV3 subprocess, must work with them. Tests: Task 8 and Task 13.
2. **Shuttle invisible for long stretches, or for the whole clip.** Expected:
   - no crash
   - no hit inferred across a tracking gap
   - a valid empty shots table and zero-hit stats

   Tests: Task 10 and Task 11.
3. **Extra people in view** (umpire, line judges, coach) or a player missing. Expected: only on-court people are kept, and a missing player becomes NaNs rather than an error. Tests: Task 9.
4. **Court corners clicked in any order, duplicated or collinear.** Expected: the order is normalised, or a clear error is raised. Tests: Task 7.
5. **Clips that aren't 30 fps 720p** (60 fps phone footage, odd dimensions, unusual aspect ratios). Expected:
   - time-based thresholds behave the same
   - the writer handles odd sizes
   - the court inset always fits

   Tests: Task 10 and Task 12.

## Scope notes

- Hit detection is only checked by eye in Phase 1. Measuring precision and recall against ShuttleSet hit frames needs the ShuttleSet video download and rally-cutting pipeline, so it moves to the Phase 3 plan, where that pipeline gets built.
- The Phase 1 shot classifier is the `"unknown"` placeholder (spec option).

## File Structure

```
pyproject.toml, uv.lock, .gitignore, README.md
configs/
  data.yaml                 dataset roots, amateur match ids, split params      (Task 6)
  shot_classes.yaml         raw ShuttleSet type -> merged class                 (Task 6)
  pipeline.yaml             tracker/pose/hits/render parameters                 (Task 13)
  splits/nycu.json, splits/shuttleset.json   generated, committed               (Task 6)
  calibrations/*.json       court corners for showcase clips                    (Task 15)
docs/data_audit.md          generated Phase 0 report                            (Task 6)
docs/phase1_notes.md        what works / fails on real clips                    (Task 15)
src/badminton_analyzer/
  __init__.py               __version__
  schema.py                 data contract: frames + shots tables                (Task 2)
  video_io.py               VideoMeta, read_video_meta, iter_frames             (Task 2)
  data/nycu.py              read_ball_csv, index_nycu                           (Task 3)
  data/shuttleset.py        strokes loader, class map                           (Task 4)
  data/video_audit.py       video URL availability                              (Task 5)
  data/splits.py            match-level splits                                  (Task 6)
  court.py                  geometry, calibration, homography                   (Task 7)
  tracking/base.py          ShuttleTracker protocol                             (Task 8)
  tracking/tracknetv3.py    subprocess wrapper                                  (Task 8)
  pose.py                   YOLOv8-pose, near/far assignment                    (Task 9)
  hits.py                   rule-based hit detection                            (Task 10)
  shots.py                  ShotClassifier, placeholder, shots table            (Task 11)
  stats.py                  rally stats                                         (Task 11)
  render.py                 annotated video                                     (Task 12)
  pipeline.py               config, run_pipeline, CLI                           (Task 13)
  app.py                    Gradio app                                          (Task 14)
scripts/audit_data.py       Phase 0 report + splits                             (Task 6)
scripts/calibrate_court.py  OpenCV corner-click tool                            (Task 7)
tests/__init__.py, tests/helpers.py, tests/test_*.py
```

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/badminton_analyzer/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: importable package `badminton_analyzer` with `__version__ = "0.1.0"`; `uv run pytest` works.

- [ ] **Step 1: Write `pyproject.toml`**
  - **Project table:**
    - `name = "badminton-analyzer"`, `version = "0.1.0"`, `requires-python = ">=3.12,<3.13"`
    - no `authors` field
    - setuptools build backend, src layout
  - **Dependencies:** `numpy>=1.26`, `pandas>=2.2`, `pyarrow>=15`, `opencv-python>=4.9`, `pyyaml>=6`, `imageio>=2.34`, `imageio-ffmpeg>=0.5`, `torch>=2.5`, `torchvision>=0.20`, `ultralytics>=8.3`, `gradio>=5`, `yt-dlp`.
  - **Optional `dev`:** `pytest>=8`.
  - **Script:** `badminton-analyze = "badminton_analyzer.pipeline:main"`.
  - **uv settings:** `[tool.uv.sources]` maps `torch` and `torchvision` to an explicit `[[tool.uv.index]]` named `pytorch-cu128` (`https://download.pytorch.org/whl/cu128`, `explicit = true`).
  - **pytest:** `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `markers = ["slow: needs real models or data"]`.

- [ ] **Step 2: Write `.gitignore`** with `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `data/`, `third_party/`, `outputs/`, `*.pt`, `*.mp4`, `mlruns/`, `wandb/`.

- [ ] **Step 3: Write `README.md` stub**
  - title "Badminton Rally Analyzer"
  - a two-sentence description from the spec's Goals
  - "Status: Phase 1 in progress"
  - a `## Credits` section listing the NYCU Shuttlecock Trajectory Dataset, ShuttleSet (KDD 2023), TrackNetV3 and Ultralytics YOLOv8 (AGPL-3.0)

- [ ] **Step 4: Write the failing test** `tests/test_smoke.py`

```python
import badminton_analyzer

def test_package_imports():
    assert badminton_analyzer.__version__ == "0.1.0"
```

- [ ] **Step 5: Create the environment and run the test.** Expected: FAIL with `AttributeError` until `__version__` exists.

Run: `uv python install 3.12; uv sync --extra dev; uv run pytest tests/test_smoke.py -v`

- [ ] **Step 6: Add `__version__ = "0.1.0"` to `src/badminton_analyzer/__init__.py` and re-run.** Expected: PASS.

- [ ] **Step 7: Check GPU torch.** Expected: prints `True`.

Run: `uv run python -c "import torch; print(torch.cuda.is_available())"`

- [ ] **Step 8: Commit** (no trailers)

```bash
git add pyproject.toml uv.lock .gitignore README.md src tests
git commit -m "chore: scaffold package, uv environment and test setup"
```

---

### Task 2: Data contract and video I/O

**Files:**
- Create: `src/badminton_analyzer/schema.py`, `src/badminton_analyzer/video_io.py`, `tests/helpers.py`, `tests/test_schema.py`, `tests/test_video_io.py`

**Interfaces:**
- **Produces, `schema.py`:**
  - `PLAYER_IDS = ("near", "far")`
  - `PLAYER_FIELDS = ["detected", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "court_x", "court_y", "kpts"]`
  - `FRAME_COLUMNS = ["frame", "shuttle_visible", "shuttle_x", "shuttle_y"] + [f"{p}_{f}" for p in PLAYER_IDS for f in PLAYER_FIELDS]`
  - `SHOT_COLUMNS = ["shot_idx", "hit_frame", "hitter", "shot_type", "hitter_court_x", "hitter_court_y"]`
  - `KPT_LEN = 51` (17 COCO keypoints × x, y, conf)
  - `class SchemaError(ValueError)`
  - `empty_frames_table(n_frames: int) -> pd.DataFrame`
  - `empty_shots_table() -> pd.DataFrame`
  - `validate_frames(df) -> None`
  - `validate_shots(df) -> None`
- **Column semantics:**
  - `frame` is 0..n-1 and contiguous.
  - `*_detected` and `shuttle_visible` are bool. Coordinates are float and NaN when absent.
  - `*_kpts` is a list of 51 floats, NaN-filled when the player isn't detected.
  - Image-space coordinates are pixels. `court_*` values are metres.
- **Produces, `video_io.py`:**
  - `@dataclass(frozen=True) VideoMeta(fps: float, n_frames: int, width: int, height: int)`
  - `read_video_meta(path: Path) -> VideoMeta` (raises `FileNotFoundError` if missing, `ValueError` if OpenCV can't open it)
  - `iter_frames(path: Path) -> Iterator[np.ndarray]` (BGR)
- **Produces, `tests/helpers.py`:**
  - `make_video(path, n_frames=30, size=(160, 90), fps=30.0) -> Path`. Creates parent dirs, writes mp4v, and gives each frame a different grey level.
  - `build_frames(n_frames, shuttle=None, near_bbox=None, far_bbox=None) -> pd.DataFrame`. Starts from `empty_frames_table`. `shuttle` maps frame to `(x, y)` and makes those frames visible. A bbox `(x1, y1, x2, y2)` marks that player detected on every frame.
  - `polyline(keyframes: list[tuple[int, tuple[float, float]]]) -> dict[int, tuple[float, float]]`: linear interpolation between keyframes, inclusive.
  - `CLICKS_160 = [(40, 20), (120, 20), (150, 85), (10, 85)]` and `CLICKS_720 = [(400, 200), (880, 200), (1100, 650), (180, 650)]`, both ordered far_left, far_right, near_right, near_left.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
def test_empty_frames_table_is_valid():
    df = empty_frames_table(5)
    validate_frames(df)
    assert list(df.columns) == FRAME_COLUMNS
    assert df["frame"].tolist() == [0, 1, 2, 3, 4]
    assert not df["shuttle_visible"].any() and not df["near_detected"].any()
    assert len(df.loc[0, "near_kpts"]) == KPT_LEN

def test_validate_frames_names_missing_column():
    with pytest.raises(SchemaError, match="far_court_y"):
        validate_frames(empty_frames_table(3).drop(columns=["far_court_y"]))

def test_validate_frames_rejects_non_contiguous_frames():
    with pytest.raises(SchemaError, match="contiguous"):
        validate_frames(empty_frames_table(3).drop(index=1))

def test_empty_shots_table_is_valid():
    df = empty_shots_table()
    validate_shots(df)
    assert list(df.columns) == SHOT_COLUMNS and len(df) == 0

# tests/test_video_io.py
def test_read_video_meta(tmp_path):
    meta = read_video_meta(make_video(tmp_path / "clip one.mp4", n_frames=30))
    assert (meta.n_frames, meta.width, meta.height) == (30, 160, 90)
    assert meta.fps == pytest.approx(30.0)

def test_iter_frames_yields_bgr_frames(tmp_path):
    frames = list(iter_frames(make_video(tmp_path / "a.mp4", n_frames=12)))
    assert len(frames) == 12 and frames[0].shape == (90, 160, 3)

def test_read_video_meta_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_video_meta(tmp_path / "nope.mp4")
```

- [ ] **Step 2: Run the tests.** Expected: FAIL (`ModuleNotFoundError`). Run: `uv run pytest tests/test_schema.py tests/test_video_io.py -v`
- [ ] **Step 3: Implement `schema.py`, `video_io.py` and `tests/helpers.py`** with the interfaces above.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: frames/shots data contract and video I/O"`

---

### Task 3: NYCU dataset loader

**Files:**
- Create: `src/badminton_analyzer/data/__init__.py`, `src/badminton_analyzer/data/nycu.py`, `tests/test_nycu.py`

**Interfaces:**
- **Produces:**
  - `read_ball_csv(path: Path) -> pd.DataFrame`:
    - columns `["frame", "visible", "x", "y"]`; `visible` is bool; `x` and `y` are NaN where not visible
    - source header is `Frame,Visibility,X,Y` (same format as TrackNetV3's predictions)
  - `@dataclass(frozen=True) NycuRally(match_id: str, rally_id: str, video_path: Path, label_path: Path)`
  - `index_nycu(root: Path) -> list[NycuRally]`:
    - finds `**/csv/*_ball.csv`; the video is `../video/<stem minus "_ball">.mp4`
    - `match_id` is the match folder's path relative to `root` with `/` separators (e.g. `"train/match1"`)
    - sorted by `(match_id, rally_id)`
    - labels without a video are skipped with one `UserWarning` stating the count

- [ ] **Step 1: Write the failing tests**

```python
def test_read_ball_csv_sets_invisible_to_nan(tmp_path):
    p = tmp_path / "1_01_00_ball.csv"
    p.write_text("Frame,Visibility,X,Y\n0,1,100,50\n1,0,0,0\n")
    df = read_ball_csv(p)
    assert list(df.columns) == ["frame", "visible", "x", "y"]
    assert df["visible"].tolist() == [True, False]
    assert df.loc[1, ["x", "y"]].isna().all()

def test_index_nycu_pairs_labels_with_videos(tmp_path):
    for m in ("match1", "match2"):
        (tmp_path / "train" / m / "csv").mkdir(parents=True)
        (tmp_path / "train" / m / "video").mkdir(parents=True)
        (tmp_path / "train" / m / "csv" / "1_01_00_ball.csv").write_text("Frame,Visibility,X,Y\n")
    (tmp_path / "train" / "match1" / "video" / "1_01_00.mp4").write_bytes(b"")
    with pytest.warns(UserWarning, match="1 label file"):
        rallies = index_nycu(tmp_path)
    assert [(r.match_id, r.rally_id) for r in rallies] == [("train/match1", "1_01_00")]
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_nycu.py -v`
- [ ] **Step 3: Implement `nycu.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: NYCU shuttle-label loader and rally index"`

---

### Task 4: ShuttleSet strokes loader and class mapping

**Files:**
- Create: `src/badminton_analyzer/data/shuttleset.py`, `tests/test_shuttleset.py`

**Interfaces:**
- **Consumes:** `SchemaError` (Task 2).
- **Produces:**
  - `REQUIRED_STROKE_COLUMNS = ("rally", "ball_round", "frame_num", "player", "type")`
  - `MERGED_CLASSES = ("serve", "net", "smash", "lob", "clear", "drive", "drop", "push_rush")`
  - `EXCLUDE = "_exclude"`
  - `load_shuttleset_strokes(root: Path) -> pd.DataFrame`:
    - concatenates every `**/set*.csv`
    - adds `match_id` (the parent folder name) and `set_id` (the file stem)
    - raises `SchemaError` naming the missing columns and the file
  - `load_shot_class_map(path: Path) -> dict[str, str]`: a YAML mapping of raw type to merged class. Any value outside `MERGED_CLASSES` and `EXCLUDE` raises `ValueError` naming it.
  - `apply_class_map(strokes: pd.DataFrame, class_map: dict[str, str]) -> pd.DataFrame`:
    - adds `shot_class` and drops rows mapped to `EXCLUDE`
    - raw types missing from the map raise `ValueError` listing them
- The real column names are confirmed against the downloaded files in Task 6. If they differ, update `REQUIRED_STROKE_COLUMNS` and these fixtures together.

- [ ] **Step 1: Write the failing tests**

```python
HEADER = "rally,ball_round,frame_num,player,type\n"

def test_load_strokes_adds_match_and_set(tmp_path):
    for m in ("matchA", "matchB"):
        (tmp_path / m).mkdir()
        (tmp_path / m / "set1.csv").write_text(HEADER + "1,1,120,A,short service\n1,2,150,B,net shot\n")
    df = load_shuttleset_strokes(tmp_path)
    assert len(df) == 4
    assert set(df["match_id"]) == {"matchA", "matchB"} and set(df["set_id"]) == {"set1"}

def test_load_strokes_names_missing_columns(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "set1.csv").write_text("rally,type\n1,smash\n")
    with pytest.raises(SchemaError, match="frame_num"):
        load_shuttleset_strokes(tmp_path)

def test_apply_class_map_merges_and_excludes():
    strokes = pd.DataFrame({"type": ["smash", "wrist smash", "unknown"]})
    out = apply_class_map(strokes, {"smash": "smash", "wrist smash": "smash", "unknown": EXCLUDE})
    assert out["shot_class"].tolist() == ["smash", "smash"]

def test_apply_class_map_rejects_unmapped_types():
    with pytest.raises(ValueError, match="lob"):
        apply_class_map(pd.DataFrame({"type": ["lob"]}), {"smash": "smash"})

def test_load_shot_class_map_rejects_unknown_class(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("smash: slam\n")
    with pytest.raises(ValueError, match="slam"):
        load_shot_class_map(p)
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_shuttleset.py -v`
- [ ] **Step 3: Implement `shuttleset.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: ShuttleSet strokes loader and shot-class mapping"`

---

### Task 5: Video availability audit

**Files:**
- Create: `src/badminton_analyzer/data/video_audit.py`, `tests/test_video_audit.py`

**Interfaces:**
- **Produces:**
  - `@dataclass(frozen=True) VideoStatus(match_id: str, url: str, available: bool, reason: str)`
  - `load_match_urls(match_csv: Path, id_column: str, url_column: str) -> list[tuple[str, str]]`: empty URL cells become `""`; a missing column raises `SchemaError`.
  - `check_urls(items, checker: Callable[[str], tuple[bool, str]]) -> list[VideoStatus]`: an empty URL gives `available=False, reason="no url"` without calling the checker.
  - `yt_dlp_checker(url: str) -> tuple[bool, str]`:
    - calls `yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}).extract_info(url, download=False)`
    - returns `(True, "")` on success
    - catches `yt_dlp.utils.DownloadError` and returns `(False, <first line of message>)`
    - needs the network and isn't unit-tested

- [ ] **Step 1: Write the failing tests**

```python
def test_check_urls_uses_checker():
    fake = lambda url: (True, "") if url.endswith("ok") else (False, "Video unavailable")
    res = check_urls([("m1", "https://y/ok"), ("m2", "https://y/gone")], fake)
    assert [(r.match_id, r.available) for r in res] == [("m1", True), ("m2", False)]
    assert res[1].reason == "Video unavailable"

def test_check_urls_marks_missing_url_without_calling_checker():
    def boom(url): raise AssertionError("should not be called")
    assert check_urls([("m1", "")], boom)[0].reason == "no url"

def test_load_match_urls(tmp_path):
    p = tmp_path / "match.csv"
    p.write_text("video,url\nm1,https://a\nm2,\n")
    assert load_match_urls(p, "video", "url") == [("m1", "https://a"), ("m2", "")]

def test_load_match_urls_missing_column(tmp_path):
    p = tmp_path / "match.csv"
    p.write_text("video\nm1\n")
    with pytest.raises(SchemaError, match="url"):
        load_match_urls(p, "video", "url")
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_video_audit.py -v`
- [ ] **Step 3: Implement `video_audit.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: ShuttleSet video availability checker"`

---

### Task 6: Match-level splits, data download and Phase 0 audit

**Files:**
- Create: `src/badminton_analyzer/data/splits.py`, `tests/test_splits.py`, `scripts/audit_data.py`, `configs/data.yaml`, `configs/shot_classes.yaml`
- Generated: `docs/data_audit.md`, `configs/splits/nycu.json`, `configs/splits/shuttleset.json`

**Interfaces:**
- **Consumes:** `index_nycu`, `read_ball_csv` (Task 3); `load_shuttleset_strokes`, `load_shot_class_map`, `apply_class_map` (Task 4); `load_match_urls`, `check_urls`, `yt_dlp_checker` (Task 5).
- **Produces:**
  - `make_match_splits(match_ids: Sequence[str], val_frac: float, test_frac: float, seed: int) -> dict[str, list[str]]`:
    - keys `"train"`, `"val"`, `"test"`, each list sorted
    - sorts the ids, then shuffles with `random.Random(seed)`
    - `n_test = max(1, round(n * test_frac))`, `n_val = max(1, round(n * val_frac))`
    - raises `ValueError` if `n < 3`
  - `save_splits(splits: dict, path: Path) -> None` and `load_splits(path: Path) -> dict` (JSON, indent 2).
  - The split files are what later phases read. `nycu.json` also has an `"amateur_test"` key.

- [ ] **Step 1: Write the failing tests**

```python
IDS = [f"m{i}" for i in range(20)]

def test_splits_are_disjoint_and_complete():
    s = make_match_splits(IDS, val_frac=0.15, test_frac=0.15, seed=42)
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(IDS)
    assert (len(s["val"]), len(s["test"])) == (3, 3)

def test_splits_ignore_input_order():
    assert make_match_splits(IDS, 0.15, 0.15, 42) == make_match_splits(IDS[::-1], 0.15, 0.15, 42)

def test_splits_change_with_seed():
    assert make_match_splits(IDS, 0.15, 0.15, 1) != make_match_splits(IDS, 0.15, 0.15, 2)

def test_splits_need_three_matches():
    with pytest.raises(ValueError):
        make_match_splits(["a", "b"], 0.15, 0.15, 42)

def test_save_load_roundtrip(tmp_path):
    s = make_match_splits(IDS, 0.15, 0.15, 42)
    save_splits(s, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == s
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_splits.py -v`
- [ ] **Step 3: Implement `splits.py`, then run the tests.** Expected: PASS.

- [ ] **Step 4: Download the data** (this may need the user to download by hand; stop and ask if a link needs a login or browser).
  - **NYCU:** follow the download link on `https://hackmd.io/@TUIK/rJkRW54cU` and unpack into `data/raw/nycu/`.
  - **ShuttleSet:** get the ShuttleSet folder from `https://github.com/wywyWang/CoachAI-Projects` (sparse clone or download) into `data/raw/shuttleset/`.
  - **Check the real files:**
    - the NYCU CSV header is `Frame,Visibility,X,Y`
    - the stroke CSVs contain `REQUIRED_STROKE_COLUMNS`
    - find the match metadata file and its match-id and URL columns

    If anything differs, fix Tasks 3–5's constants and fixtures first, re-run `uv run pytest`, and commit.

- [ ] **Step 5: Write `configs/data.yaml`**

```yaml
nycu:
  root: data/raw/nycu
  amateur_match_ids: []        # fill in Step 7
shuttleset:
  root: data/raw/shuttleset
  match_csv: data/raw/shuttleset/<path found in Step 4>
  match_id_column: <found in Step 4>   # must equal the stroke folders' names
  url_column: <found in Step 4>
  class_map: configs/shot_classes.yaml
splits: {seed: 42, val_frac: 0.15, test_frac: 0.15}
```

- [ ] **Step 6: Implement and run `scripts/audit_data.py --config configs/data.yaml --out docs/data_audit.md`**

  **It writes a markdown report with:**
  - **NYCU:** a per-match table of rallies, frames and the share of frames where the shuttle is visible.
  - **ShuttleSet:** total matches and strokes, and raw `type` counts. Merged-class counts are included only if the class map loads.
  - **Video availability:** a table of match id, available yes/no and reason, plus a line `Available videos: X/44`.

  **It also writes the split files:**
  - `configs/splits/nycu.json`: splits over the NYCU match ids excluding `amateur_match_ids`, plus `"amateur_test": amateur_match_ids`.
  - `configs/splits/shuttleset.json`: splits over the ShuttleSet matches whose video is available.

  The first run is expected to fail on the class map. Go to Step 7.

- [ ] **Step 7: Fill the configs from the first report, then re-run Step 6**
  - **`amateur_match_ids`:** the 3 amateur NYCU matches. The NYCU page says they are the last three; confirm by opening one video from each.
  - **`configs/shot_classes.yaml`:** map every raw `type` value in the report (they may be Chinese) to a merged class, following the ShuttleSet paper's English type table:
    - serve ← short service, long service
    - net ← net shot, return net, cross-court net shot
    - smash ← smash, wrist smash
    - lob ← lob, defensive return lob
    - clear ← clear
    - drive ← drive, driven flight, back-court drive, defensive return drive
    - drop ← drop, passive drop
    - push_rush ← push, rush
    - unknown/other → `_exclude`
  - Re-run. Expected: the report has merged-class counts and an `Available videos: X/44` line, and both split files exist.

- [ ] **Step 8: Commit** with `git add src tests scripts configs docs/data_audit.md` then `git commit -m "feat: match-level splits and Phase 0 data audit"`

- [ ] **Step 9: CHECKPOINT. Report to the user before continuing:**
  - NYCU totals
  - merged-class counts, flagging any class under 500 strokes
  - available videos X/44

  If X < 25, stop: the spec says to pick a fallback before Phase 3. Phase 1 (Tasks 7–15) doesn't depend on that decision, so continue it once the user has seen the numbers.

---

### Task 7: Court geometry and calibration

**Files:**
- Create: `src/badminton_analyzer/court.py`, `scripts/calibrate_court.py`, `tests/test_court.py`

**Interfaces:**
- **Produces:**
  - Constants: `COURT_LENGTH_M = 13.40`, `COURT_WIDTH_M = 6.10`, `SINGLES_WIDTH_M = 5.18`, `NET_Y_M = 6.70`, `SHORT_SERVICE_FROM_NET_M = 1.98`
  - `@dataclass(frozen=True) CourtCalibration(image_points: np.ndarray, homography: np.ndarray)`:
    - `image_points` is (4, 2), ordered far_left, far_right, near_right, near_left
    - `homography` is 3×3, image → court
  - `order_corners(points) -> np.ndarray`:
    - the two smallest-y points are the far corners, ordered by x
    - the other two are the near corners: near_right (larger x), then near_left
    - raises `ValueError` mentioning "4" if not exactly 4 points
    - raises `ValueError` mentioning "corners" if the shoelace area is under 100 px²
  - `calibrate(points: Sequence[tuple[float, float]]) -> CourtCalibration`: uses `cv2.getPerspectiveTransform` onto `[(0,0), (6.10,0), (6.10,13.40), (0,13.40)]`.
  - `image_to_court(cal, pts: np.ndarray) -> np.ndarray`: (N, 2) → (N, 2); NaN rows stay NaN.
  - `save_calibration(cal, path)` and `load_calibration(path) -> CourtCalibration`: JSON `{"image_points": [[x, y], ...]}`; the homography is recomputed on load.

- [ ] **Step 1: Write the failing tests**

```python
COURT = [[0, 0], [6.10, 0], [6.10, 13.40], [0, 13.40]]

def test_corners_map_to_court_corners():
    out = image_to_court(calibrate(CLICKS_720), np.array(CLICKS_720, dtype=float))
    np.testing.assert_allclose(out, COURT, atol=1e-6)

def test_click_order_does_not_matter():
    shuffled = [CLICKS_720[i] for i in (2, 0, 3, 1)]
    np.testing.assert_allclose(calibrate(shuffled).homography, calibrate(CLICKS_720).homography)

def test_collinear_corners_rejected():
    with pytest.raises(ValueError, match="corners"):
        calibrate([(0, 0), (50, 0), (100, 0), (150, 0)])

def test_wrong_number_of_points_rejected():
    with pytest.raises(ValueError, match="4"):
        calibrate(CLICKS_720[:3])

def test_nan_points_pass_through():
    out = image_to_court(calibrate(CLICKS_720), np.array([[np.nan, np.nan], [640.0, 420.0]]))
    assert np.isnan(out[0]).all() and np.isfinite(out[1]).all()

def test_calibration_roundtrip(tmp_path):
    cal = calibrate(CLICKS_720)
    save_calibration(cal, tmp_path / "c d" / "cal.json")
    np.testing.assert_allclose(load_calibration(tmp_path / "c d" / "cal.json").homography, cal.homography)
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_court.py -v`
- [ ] **Step 3: Implement `court.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Implement `scripts/calibrate_court.py VIDEO --out CAL.json`**
  - Shows the first frame in an OpenCV window and records left clicks, drawing a dot and an index at each.
  - `r` resets the points. `Enter` with 4 points saves via `save_calibration(calibrate(points), out)`. `Esc` quits without saving.
  - **Manual check:** run it on any NYCU rally and confirm the JSON has 4 points.

- [ ] **Step 6: Commit** with `git commit -m "feat: court calibration and image-to-court homography"`

---

### Task 8: Shuttle tracker protocol and TrackNetV3 wrapper

**Files:**
- Create: `src/badminton_analyzer/tracking/__init__.py`, `tracking/base.py`, `tracking/tracknetv3.py`, `tests/test_tracknetv3.py`

**Interfaces:**
- **Consumes:** `read_ball_csv` (Task 3).
- **Produces:**
  - `class ShuttleTracker(Protocol): def track(self, video_path: Path) -> pd.DataFrame`. It returns `read_ball_csv`'s format: `frame`, `visible`, `x`, `y`.
  - `run_subprocess(cmd: list[str], cwd: Path) -> None`:
    - calls `subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)`
    - on `CalledProcessError`, raises `RuntimeError` with the last 20 lines of stderr
  - `@dataclass TrackNetV3Tracker(repo_dir: Path, tracknet_ckpt: Path, inpaintnet_ckpt: Path, work_dir: Path, batch_size: int = 16, runner: Callable[[list[str], Path], None] = run_subprocess)`:
    - `build_command(video_path: Path, save_dir: Path) -> list[str]` returns `[sys.executable, "predict.py", "--video_file", str(video_path), "--tracknet_file", str(tracknet_ckpt), "--inpaintnet_file", str(inpaintnet_ckpt), "--save_dir", str(save_dir), "--batch_size", str(batch_size)]`
    - `track(video_path)` runs in `cwd=repo_dir` with `save_dir = work_dir / video_path.stem`, then reads `save_dir / f"{video_path.stem}_ball.csv"`
    - if that file is missing, it raises `RuntimeError` whose message contains `_ball.csv`

- [ ] **Step 1: Write the failing tests**

```python
def fake_runner(rows: str):
    def run(cmd, cwd):
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        stem = Path(cmd[cmd.index("--video_file") + 1]).stem
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{stem}_ball.csv").write_text("Frame,Visibility,X,Y\n" + rows)
    return run

def make_tracker(tmp_path, runner):
    return TrackNetV3Tracker(tmp_path / "repo", tmp_path / "t.pt", tmp_path / "i.pt", tmp_path / "work dir", runner=runner)

def test_tracker_returns_ball_table(tmp_path):
    df = make_tracker(tmp_path, fake_runner("0,1,10,20\n1,0,0,0\n")).track(tmp_path / "my clips" / "rally 1.mp4")
    assert df["visible"].tolist() == [True, False]

def test_command_keeps_paths_with_spaces_intact(tmp_path):
    cmd = make_tracker(tmp_path, fake_runner("")).build_command(Path("C:/a b/rally 1.mp4"), Path("C:/out dir"))
    assert "C:/a b/rally 1.mp4" in [c.replace("\\", "/") for c in cmd]

def test_missing_output_raises(tmp_path):
    with pytest.raises(RuntimeError, match="_ball.csv"):
        make_tracker(tmp_path, lambda cmd, cwd: None).track(tmp_path / "x.mp4")
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_tracknetv3.py -v`
- [ ] **Step 3: Implement `base.py` and `tracknetv3.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Set up the real TrackNetV3**
  - `git clone https://github.com/qaz812345/TrackNetV3 third_party/TrackNetV3`, and note the commit hash with `git -C third_party/TrackNetV3 rev-parse HEAD`.
  - Download the checkpoints its README links into `third_party/TrackNetV3/ckpts/`.
  - Install any packages from its requirements that our environment lacks with `uv add --optional tracknetv3 <pkg>`. Never downgrade torch.
  - Check that `predict.py`'s flag names match `build_command`. If they don't, fix `build_command` and its test.
  - **Smoke run** (Python one-liner or REPL):
    - `TrackNetV3Tracker(...).track(<one NYCU rally video>)`
    - Expected: a table with one row per frame, mostly `visible == True`.
- [ ] **Step 6: Commit.** Add the TrackNetV3 commit hash to README Credits, then `git commit -m "feat: shuttle tracker protocol and TrackNetV3 wrapper"`

---

### Task 9: Player pose and near/far assignment

**Files:**
- Create: `src/badminton_analyzer/pose.py`, `tests/test_pose.py`

**Interfaces:**
- **Consumes:** `CourtCalibration`, `image_to_court`, `NET_Y_M`, `COURT_WIDTH_M`, `COURT_LENGTH_M` (Task 7); `empty_frames_table`, `KPT_LEN` (Task 2); `iter_frames`, `read_video_meta` (Task 2).
- **Produces:**
  - `@dataclass(frozen=True) PersonDetection(bbox: np.ndarray, kpts: np.ndarray, conf: float)`, where `bbox` is (4,) x1y1x2y2 and `kpts` is (17, 3).
  - `foot_point(det) -> np.ndarray`: the midpoint of the ankles (COCO 15, 16) when both have conf ≥ 0.3, otherwise the bbox bottom-centre.
  - `assign_players(dets, cal, side_margin_m: float, end_margin_m: float) -> dict[str, PersonDetection | None]`:
    - keeps people whose foot point maps inside `[-side, 6.10+side] × [-end, 13.40+end]`
    - `court_y < NET_Y_M` is far, otherwise near
    - keeps the highest-conf person per half
  - `class PoseEstimator(Protocol): def detect(self, frame: np.ndarray) -> list[PersonDetection]`
  - `class YoloPoseEstimator(weights: str = "yolov8m-pose.pt", conf: float = 0.3)`:
    - `detect` uses `YOLO(weights).predict(frame, conf=conf, verbose=False)[0]`
    - reads `boxes.xyxy`, `boxes.conf` and `keypoints.data`
  - `extract_players(video_path, estimator, cal, side_margin_m, end_margin_m) -> pd.DataFrame`:
    - a frames table (`empty_frames_table(n)` with every `near_*`/`far_*` column filled), one row per decoded frame
    - for a detected player: `{p}_detected = True`, plus bbox, foot point, court x/y and the flattened kpts

- [ ] **Step 1: Write the failing tests** (`det(foot, conf, ankle_conf=0.9)` is a local helper: a 60×160 bbox standing on `foot`, ankles at `foot ± (10, 0)`).

```python
CAL = calibrate(CLICKS_720)
def foot(d): return tuple(np.round(foot_point(d)).astype(int))

def test_assigns_near_far_and_drops_umpire():
    out = assign_players([det((640, 600), .9), det((640, 260), .8), det((60, 420), .95)], CAL, 0.75, 2.0)
    assert foot(out["near"]) == (640, 600) and foot(out["far"]) == (640, 260)

def test_keeps_most_confident_per_half():
    out = assign_players([det((600, 600), .5), det((700, 600), .9)], CAL, 0.75, 2.0)
    assert foot(out["near"]) == (700, 600) and out["far"] is None

def test_foot_point_falls_back_to_bbox_bottom():
    d = det((640, 600), .9, ankle_conf=0.1)
    np.testing.assert_allclose(foot_point(d), [(d.bbox[0] + d.bbox[2]) / 2, d.bbox[3]])

def test_extract_players_fills_every_frame(tmp_path):
    class NearOnly:
        def detect(self, frame): return [det((80, 86), .9)]
    df = extract_players(make_video(tmp_path / "v.mp4", n_frames=10), NearOnly(), calibrate(CLICKS_160), 0.75, 2.0)
    assert len(df) == 10 and df["near_detected"].all() and not df["far_detected"].any()
    assert df["far_court_x"].isna().all() and (df["near_court_y"] > 6.70).all()
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_pose.py -v`
- [ ] **Step 3: Implement `pose.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: YOLOv8-pose player extraction with on-court near/far assignment"`

---

### Task 10: Rule-based hit detection

**Files:**
- Create: `src/badminton_analyzer/hits.py`, `tests/test_hits.py`

**Interfaces:**
- **Consumes:** the frames table (Task 2).
- **Produces:**
  - `@dataclass(frozen=True) HitParams(min_gap_s=0.25, min_turn_deg=45.0, max_hitter_dist_frac=0.12, smooth_window=3, max_gap_s=0.1, serve_speed_frac=0.2, bbox_top_extend=0.3)`. `serve_speed_frac` is in frame heights per second, so it doesn't depend on fps.
  - `detect_hits(frames: pd.DataFrame, fps: float, frame_height: int, params: HitParams) -> pd.DataFrame`, columns `["hit_frame", "hitter"]`, sorted by frame. When there are no hits it is empty but keeps both columns.

**Algorithm** (the tests don't fully pin this down):
1. **Segments:** split visible shuttle points into segments wherever the frame gap is more than `round(max_gap_s * fps)`. Smooth positions within each segment with a centred moving average of `smooth_window`.
2. **Turns:** a turn candidate is an interior point of a segment where the angle between the incoming and outgoing velocity is at least `min_turn_deg`. Its score is that angle.
3. **Serve candidate:** the first frame of the first segment where shuttle speed (step length × fps) exceeds `serve_speed_frac * frame_height`. Its score is 180.
4. **Hitter:**
   - For each candidate, compute the distance from the shuttle to each detected player's bbox. Extend the bbox upward by `bbox_top_extend × bbox height`; the distance is 0 if the shuttle is inside.
   - The hitter is the closer player. Drop the candidate if that distance exceeds `max_hitter_dist_frac * frame_height`.
5. **Clustering:** group candidates less than `round(min_gap_s * fps)` frames apart and keep the highest score in each group.
6. **Alternation:** in time order, if a hit has the same hitter as the previous kept hit, keep whichever has the higher score.

- [ ] **Step 1: Write the failing tests** (720p geometry, so `frame_height = 720`)

```python
NEAR, FAR = (600, 450, 680, 610), (610, 150, 670, 270)
def hits_of(shuttle, n, fps=30.0, near=NEAR, far=FAR):
    h = detect_hits(build_frames(n, shuttle, near, far), fps, 720, HitParams())
    return list(zip(h["hit_frame"], h["hitter"]))

def test_serve_and_return_detected():
    got = hits_of(polyline([(0, (640, 200)), (30, (640, 470)), (60, (640, 200))]), 61)
    assert got[0] == (0, "far") and got[1][1] == "near" and abs(got[1][0] - 30) <= 1 and len(got) == 2

def test_turn_far_from_players_is_ignored():
    shuttle = polyline([(0, (640, 200)), (30, (640, 470)), (60, (640, 200))])
    assert hits_of(shuttle, 61, near=(100, 450, 180, 610), far=(100, 150, 180, 270)) == []

def test_all_invisible_returns_empty_table():
    h = detect_hits(build_frames(30, None, NEAR, FAR), 30.0, 720, HitParams())
    assert h.empty and list(h.columns) == ["hit_frame", "hitter"]

def test_no_hit_inferred_across_tracking_gap():
    down = polyline([(0, (640, 200)), (30, (640, 470))])
    up = polyline([(51, (640, 470)), (80, (640, 200))])
    assert hits_of({**down, **up}, 81) == [(0, "far")]

def test_same_hitter_twice_keeps_sharper_turn():
    shuttle = polyline([(0, (640, 200)), (30, (640, 470)), (45, (660, 430)), (60, (640, 470)), (90, (640, 200))])
    got = hits_of(shuttle, 91)
    assert got[0] == (0, "far") and len(got) == 2 and got[1][1] == "near" and abs(got[1][0] - 45) <= 1

def test_thresholds_scale_with_fps():
    got = hits_of(polyline([(0, (640, 200)), (60, (640, 470)), (120, (640, 200))]), 121, fps=60.0)
    assert len(got) == 2 and abs(got[1][0] - 60) <= 2
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_hits.py -v`
- [ ] **Step 3: Implement `hits.py`** following the algorithm above.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: rule-based hit detection with hitter assignment"`

---

### Task 11: Shot classifier placeholder, shots table and rally stats

**Files:**
- Create: `src/badminton_analyzer/shots.py`, `src/badminton_analyzer/stats.py`, `tests/test_shots.py`, `tests/test_stats.py`

**Interfaces:**
- **Consumes:** frames and shots tables, `validate_shots`, `empty_shots_table` (Task 2); `detect_hits` output columns (Task 10).
- **Produces:**
  - `class ShotClassifier(Protocol): def classify(self, video_path: Path, frames: pd.DataFrame, hits: pd.DataFrame) -> list[str]`, returning one label per hit row.
  - `class UnknownShotClassifier`: returns `["unknown"] * len(hits)`.
  - `build_shots_table(frames, hits, labels) -> pd.DataFrame` in the `SHOT_COLUMNS` shape:
    - `shot_idx` is 0..k-1
    - `hitter_court_x` and `hitter_court_y` are the hitter's court position at `hit_frame` (NaN if not detected)
  - `rally_stats(frames, shots, fps: float, max_step_m: float = 0.5) -> dict`, JSON-safe with NaN written as `None`, keys:
    - `n_frames`, `fps`, `duration_s`, `n_shots`
    - `shot_counts` (type → count)
    - `shots`: a list of `{shot_idx, hit_frame, time_s, hitter, shot_type, court_x, court_y}`
    - `players`: `{"near" | "far": {"distance_m", "detected_frac"}}`
  - **Distance covered:**
    - smooth court x/y with a centred rolling median of window 5 over detected frames
    - sum the step lengths between consecutive detected frames
    - skip steps over `max_step_m` and steps across gaps

- [ ] **Step 1: Write the failing tests**

```python
def walking_frames(n=91):
    f = build_frames(n)
    f["near_detected"] = True
    f["near_court_x"] = 3.0
    f["near_court_y"] = np.linspace(8.0, 11.0, n)
    return f

def test_distance_covered_straight_walk():
    s = rally_stats(walking_frames(), empty_shots_table(), fps=30.0)
    assert s["players"]["near"]["distance_m"] == pytest.approx(3.0, abs=0.05)

def test_detection_spike_is_ignored():
    f = walking_frames()
    f.loc[45, "near_court_y"] += 5.0
    assert rally_stats(f, empty_shots_table(), 30.0)["players"]["near"]["distance_m"] == pytest.approx(3.0, abs=0.05)

def test_no_shots_is_json_safe():
    s = rally_stats(build_frames(30), empty_shots_table(), 30.0)
    assert s["n_shots"] == 0 and s["shot_counts"] == {} and s["players"]["far"]["distance_m"] == 0.0
    json.dumps(s, allow_nan=False)

def test_build_shots_table_reads_hitter_position():
    f = build_frames(40)
    f.loc[30, ["near_detected", "near_court_x", "near_court_y"]] = [True, 3.0, 11.0]
    shots = build_shots_table(f, pd.DataFrame({"hit_frame": [30], "hitter": ["near"]}), ["unknown"])
    validate_shots(shots)
    assert shots.iloc[0][["hitter_court_x", "hitter_court_y"]].tolist() == [3.0, 11.0]

def test_stats_shot_times():
    shots = build_shots_table(build_frames(60), pd.DataFrame({"hit_frame": [0, 30], "hitter": ["far", "near"]}), ["unknown"] * 2)
    s = rally_stats(build_frames(60), shots, 30.0)
    assert s["shot_counts"] == {"unknown": 2} and [x["time_s"] for x in s["shots"]] == [0.0, 1.0]
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_shots.py tests/test_stats.py -v`
- [ ] **Step 3: Implement `shots.py` and `stats.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: shot placeholder, shots table and rally stats"`

---

### Task 12: Annotated video renderer

**Files:**
- Create: `src/badminton_analyzer/render.py`, `tests/test_render.py`

**Interfaces:**
- **Consumes:** court constants (Task 7), frames and shots tables (Task 2), `iter_frames` and `read_video_meta` (Task 2).
- **Produces:**
  - `@dataclass(frozen=True) RenderParams(trail_length: int = 8, label_duration_s: float = 0.7, inset_width_frac: float = 0.18)`
  - `inset_size(frame_w: int, frame_h: int, width_frac: float) -> tuple[int, int]`:
    - width is `round(frame_w * width_frac)` and height is `round(width * 13.40 / 6.10)`
    - if the height exceeds `0.9 * frame_h`, scale both down to fit
  - `draw_court_inset(width_px: int, players: dict[str, tuple[float, float] | None]) -> np.ndarray`:
    - shape is `(round(width_px * 13.40 / 6.10), width_px, 3)`
    - draws the outer lines, singles sidelines, net, short service lines and centre lines, plus a dot per player (near and far in different colours)
  - `draw_frame(frame, frame_idx, frames, shots, fps, params) -> np.ndarray`:
    - returns a copy; the input is never modified
    - **shuttle trail:** the last `trail_length` frames, connecting only consecutive visible frames, plus a filled circle (radius 3 px) at the current position when visible
    - **players:** COCO-17 skeletons, near and far in different colours
    - **shot labels:** shown above the hitter's bbox from `hit_frame` for `label_duration_s`
    - **inset:** top-right, 8 px margin
    - frames missing from the table are drawn without overlay
  - `render_video(video_path, frames, shots, out_path, params) -> Path`:
    - writes H.264 via `imageio.v2.get_writer(str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)`
    - converts BGR to RGB and pads each frame by 1 px to even dimensions where needed
    - creates parent dirs

- [ ] **Step 1: Write the failing tests**

```python
def test_inset_shape_follows_court_aspect():
    assert draw_court_inset(100, {"near": (3.05, 10.0), "far": None}).shape == (round(100 * 13.40 / 6.10), 100, 3)

def test_inset_fits_short_wide_frames():
    w, h = inset_size(400, 90, 0.18)
    assert h <= 81 and w / h == pytest.approx(6.10 / 13.40, rel=0.05)

def test_draw_frame_marks_shuttle_without_touching_input():
    frame = np.zeros((90, 160, 3), np.uint8)
    out = draw_frame(frame, 9, build_frames(10, {i: (60, 45) for i in range(10)}), empty_shots_table(), 30.0, RenderParams())
    assert out.shape == frame.shape and out[45, 60].any() and not frame.any()

def test_render_video_keeps_frame_count_with_odd_size(tmp_path):
    src = make_video(tmp_path / "in clip.mp4", n_frames=20, size=(161, 91))
    out = render_video(src, build_frames(20), empty_shots_table(), tmp_path / "out dir" / "annotated.mp4", RenderParams())
    assert read_video_meta(out).n_frames == 20
```

- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_render.py -v`
- [ ] **Step 3: Implement `render.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -m "feat: annotated video renderer with court inset"`

---

### Task 13: Pipeline orchestration, config and CLI

**Files:**
- Create: `src/badminton_analyzer/pipeline.py`, `configs/pipeline.yaml`, `tests/test_pipeline.py`

**Interfaces:**
- **Consumes:**
  - `ShuttleTracker`, `TrackNetV3Tracker` (Task 8)
  - `PoseEstimator`, `YoloPoseEstimator`, `extract_players` (Task 9)
  - `HitParams`, `detect_hits` (Task 10)
  - `ShotClassifier`, `UnknownShotClassifier`, `build_shots_table`, `rally_stats` (Task 11)
  - `RenderParams`, `render_video` (Task 12)
  - `load_calibration` (Task 7), `validate_frames`, `validate_shots` (Task 2)
- **Produces:**
  - `@dataclass(frozen=True) PipelineConfig(tracker: dict, pose: dict, hits: HitParams, render: RenderParams)`
  - `load_pipeline_config(path: Path) -> PipelineConfig`
  - `@dataclass(frozen=True) PipelineOutputs(frames_path, shots_path, stats_path, video_path)`, whose files are `frames.parquet`, `shots.parquet`, `stats.json` and `annotated.mp4` in `out_dir`
  - `run_pipeline(video_path, calibration, out_dir, tracker, pose, shot_classifier, cfg) -> PipelineOutputs`:
    - tracker output is renamed to `shuttle_*` and merged on `frame` into the players table from `extract_players`
    - tracker rows beyond the decoded frame count are dropped, and missing rows count as invisible
  - `build_default_components(cfg) -> tuple[ShuttleTracker, PoseEstimator, ShotClassifier]`
  - `main(argv: list[str] | None = None) -> int`:
    - takes `video`, `--calib PATH` (required), `--out DIR` (default `outputs/<video stem>`) and `--config` (default `configs/pipeline.yaml`)
    - returns 0 on success

- [ ] **Step 1: Write `configs/pipeline.yaml`**

```yaml
tracker:
  kind: tracknetv3
  repo_dir: third_party/TrackNetV3
  tracknet_ckpt: third_party/TrackNetV3/ckpts/TrackNet_best.pt
  inpaintnet_ckpt: third_party/TrackNetV3/ckpts/InpaintNet_best.pt
  work_dir: outputs/_tracknet
  batch_size: 16
pose: {weights: yolov8m-pose.pt, conf: 0.3, side_margin_m: 0.75, end_margin_m: 2.0}
hits: {min_gap_s: 0.25, min_turn_deg: 45.0, max_hitter_dist_frac: 0.12, smooth_window: 3, max_gap_s: 0.1, serve_speed_frac: 0.2, bbox_top_extend: 0.3}
render: {trail_length: 8, label_duration_s: 0.7, inset_width_frac: 0.18}
```

(Use the checkpoint filenames actually downloaded in Task 8.)

- [ ] **Step 2: Write the failing tests.** Put `CONFIG`, `FakeTracker` and `FakePose` in `tests/helpers.py`, since Task 14 reuses them. The fakes are based on the `CLICKS_160` geometry.

```python
# tests/helpers.py
CONFIG = Path(__file__).parents[1] / "configs" / "pipeline.yaml"

class FakeTracker:
    def __init__(self, n=60): self.n = n
    def track(self, video_path):
        pts = polyline([(0, (80, 25)), (30, (80, 70)), (59, (80, 25))])
        return pd.DataFrame({"frame": range(self.n), "visible": [i in pts for i in range(self.n)],
                             "x": [pts.get(i, (np.nan,))[0] for i in range(self.n)],
                             "y": [pts.get(i, (np.nan, np.nan))[1] for i in range(self.n)]})

class FakePose:
    def detect(self, frame):
        return [PersonDetection(np.array([70, 55, 90, 88.0]), np.zeros((17, 3)), 0.9),
                PersonDetection(np.array([72, 12, 88, 34.0]), np.zeros((17, 3)), 0.9)]

# tests/test_pipeline.py
def run(tmp_path, tracker):
    video = make_video(tmp_path / "my clips" / "rally 1.mp4", n_frames=60)
    return run_pipeline(video, calibrate(CLICKS_160), tmp_path / "out dir", tracker, FakePose(),
                        UnknownShotClassifier(), load_pipeline_config(CONFIG))

def test_pipeline_writes_all_outputs(tmp_path):
    out = run(tmp_path, FakeTracker())
    validate_frames(pd.read_parquet(out.frames_path))
    validate_shots(pd.read_parquet(out.shots_path))
    stats = json.loads(out.stats_path.read_text())
    assert stats["n_frames"] == 60 and stats["n_shots"] >= 1
    assert read_video_meta(out.video_path).n_frames == 60

def test_tracker_output_longer_than_video_is_clipped(tmp_path):
    out = run(tmp_path, FakeTracker(n=70))
    assert len(pd.read_parquet(out.frames_path)) == 60

def test_cli_main(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "build_default_components",
                        lambda cfg: (FakeTracker(), FakePose(), UnknownShotClassifier()))
    video = make_video(tmp_path / "v.mp4", n_frames=60)
    save_calibration(calibrate(CLICKS_160), tmp_path / "cal.json")
    rc = pipeline.main([str(video), "--calib", str(tmp_path / "cal.json"), "--out", str(tmp_path / "o"), "--config", str(CONFIG)])
    assert rc == 0 and (tmp_path / "o" / "annotated.mp4").exists()
```

- [ ] **Step 3: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_pipeline.py -v`
- [ ] **Step 4: Implement `pipeline.py`.**
- [ ] **Step 5: Run the full suite.** Expected: all PASS. Run: `uv run pytest -v`
- [ ] **Step 6: Commit** with `git commit -m "feat: end-to-end pipeline, config and CLI"`

---

### Task 14: Local Gradio app

**Files:**
- Create: `src/badminton_analyzer/app.py`, `tests/test_app.py`

**Interfaces:**
- **Consumes:** `load_pipeline_config`, `build_default_components`, `run_pipeline` (Task 13); `calibrate` (Task 7); `iter_frames` (Task 2).
- **Produces:**
  - `add_corner(points: list[tuple[int, int]], xy: tuple[int, int]) -> list[tuple[int, int]]`: returns a new list and ignores clicks after the 4th.
  - `draw_corner_markers(frame_rgb: np.ndarray, points) -> np.ndarray`: returns a copy.
  - `first_frame(video_path: str) -> np.ndarray`: RGB.
  - `get_components(config_path: str)`: wraps `build_default_components`, cached with `functools.lru_cache`.
  - `analyze(video_path: str, points, config_path: str, output_root: Path = Path("outputs/app")) -> tuple[str | None, dict | None, str]`:
    - returns the annotated video path, the stats and a status message
    - with fewer than 4 points it returns `(None, None, f"Click all 4 court corners first ({n}/4).")`
    - otherwise it writes to `output_root / f"{YYYYmmdd-HHMMSS}-{stem}"`, and the status starts with `"Done"`
    - exceptions become a status starting with `"Error: "`
  - `build_app(config_path: str = "configs/pipeline.yaml") -> gr.Blocks`:
    - a Video upload; loading a clip shows the first frame in an Image whose `.select` event calls `add_corner` and redraws the markers
    - a "Reset corners" button and an "Analyze" button
    - outputs: Video, JSON and Markdown status
    - `python -m badminton_analyzer.app` launches it

- [ ] **Step 1: Write the failing tests**

```python
def test_add_corner_caps_at_four():
    pts = []
    for xy in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]:
        pts = add_corner(pts, xy)
    assert pts == [(1, 1), (2, 2), (3, 3), (4, 4)]

def test_draw_corner_markers_copies_input():
    img = np.zeros((90, 160, 3), np.uint8)
    out = draw_corner_markers(img, [(10, 10)])
    assert out.any() and not img.any()

def test_analyze_needs_four_corners(tmp_path):
    assert analyze(str(make_video(tmp_path / "a.mp4")), [(1, 1)], str(CONFIG), tmp_path) == \
        (None, None, "Click all 4 court corners first (1/4).")

def test_analyze_runs_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "get_components", lambda cfg: (FakeTracker(), FakePose(), UnknownShotClassifier()))
    video_out, stats, status = analyze(str(make_video(tmp_path / "a b.mp4", n_frames=60)), CLICKS_160, str(CONFIG), tmp_path)
    assert Path(video_out).exists() and stats["n_frames"] == 60 and status.startswith("Done")
```
- [ ] **Step 2: Run the tests.** Expected: FAIL. Run: `uv run pytest tests/test_app.py -v`
- [ ] **Step 3: Implement `app.py`.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Manual check.** Run `uv run python -m badminton_analyzer.app`, upload an NYCU test-split rally, click the 4 corners and press Analyze. Expected: the annotated video plays in the browser and the stats JSON shows.
- [ ] **Step 6: Commit** with `git commit -m "feat: local Gradio app with click-to-calibrate"`

---

### Task 15: Phase 1 verification on real clips and README

**Files:**
- Create: `configs/calibrations/<rally>.json` (two of them), `docs/phase1_notes.md`
- Modify: `README.md`

- [ ] **Step 1: Pick the clips.** Take one rally from `configs/splits/nycu.json` `"test"` and one from `"amateur_test"`.
- [ ] **Step 2: Calibrate each.** Run `uv run python scripts/calibrate_court.py <video> --out configs/calibrations/<match>_<rally>.json` for both.
- [ ] **Step 3: Run the CLI on both.** Run: `uv run badminton-analyze <video> --calib <cal.json> --out outputs/<match>_<rally>`

  Expected for each: `frames.parquet`, `shots.parquet`, `stats.json` and `annotated.mp4` exist, and `annotated.mp4` plays.
- [ ] **Step 4: Watch both videos and record what you see in `docs/phase1_notes.md`.** For each clip, write down:
  - Does the shuttle trail follow the shuttle?
  - Are skeletons drawn on exactly the two players, with no umpire or line judges?
  - Do the inset dots move sensibly?
  - Do the `unknown` labels appear at real hits? Count missed and extra hits by eye.
  - How long did it take to run?

  These notes become the "before" baseline for Phases 2–3.
- [ ] **Step 5: Update the README**
  - **Setup:** `uv sync`, TrackNetV3 clone and checkpoints, data download.
  - **Usage:** the CLI and the app.
  - **Pipeline:** the diagram from the spec.
  - **Status:** Phase 1 done.
  - **Credits:** TrackNetV3 commit hash, datasets, Ultralytics AGPL-3.0.
  - No AI-assistant credit.
- [ ] **Step 6: Run the full test suite.** Expected: all PASS. Run: `uv run pytest -v`
- [ ] **Step 7: Commit** with `git add configs/calibrations docs/phase1_notes.md README.md` then `git commit -m "docs: Phase 1 verification notes and README usage"`
