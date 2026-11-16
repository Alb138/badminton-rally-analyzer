"""Data contract shared by every pipeline stage.

Two tables flow through the pipeline:

* the frames table: one row per video frame, with the shuttle position and
  both players' boxes, foot points, court positions and keypoints;
* the shots table: one row per detected hit.

Image-space values are pixels, ``court_*`` values are metres, and anything
absent is NaN.
"""
import numpy as np
import pandas as pd

PLAYER_IDS = ("near", "far")
PLAYER_FIELDS = ["detected", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "court_x", "court_y", "kpts"]
FRAME_COLUMNS = ["frame", "shuttle_visible", "shuttle_x", "shuttle_y"] + [
    f"{p}_{f}" for p in PLAYER_IDS for f in PLAYER_FIELDS
]
SHOT_COLUMNS = ["shot_idx", "hit_frame", "hitter", "shot_type", "hitter_court_x", "hitter_court_y"]
KPT_LEN = 51  # 17 COCO keypoints x (x, y, conf)


class SchemaError(ValueError):
    """A table does not match the data contract."""


def empty_frames_table(n_frames: int) -> pd.DataFrame:
    data: dict[str, object] = {
        "frame": np.arange(n_frames, dtype=np.int64),
        "shuttle_visible": np.zeros(n_frames, dtype=bool),
        "shuttle_x": np.full(n_frames, np.nan),
        "shuttle_y": np.full(n_frames, np.nan),
    }
    for pid in PLAYER_IDS:
        for field in PLAYER_FIELDS:
            col = f"{pid}_{field}"
            if field == "detected":
                data[col] = np.zeros(n_frames, dtype=bool)
            elif field == "kpts":
                data[col] = [[np.nan] * KPT_LEN for _ in range(n_frames)]
            else:
                data[col] = np.full(n_frames, np.nan)
    return pd.DataFrame(data, columns=FRAME_COLUMNS)


def empty_shots_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_idx": pd.Series(dtype="int64"),
            "hit_frame": pd.Series(dtype="int64"),
            "hitter": pd.Series(dtype="object"),
            "shot_type": pd.Series(dtype="object"),
            "hitter_court_x": pd.Series(dtype="float64"),
            "hitter_court_y": pd.Series(dtype="float64"),
        },
        columns=SHOT_COLUMNS,
    )


def _require_columns(df: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{table} table is missing columns: {', '.join(missing)}")


def validate_frames(df: pd.DataFrame) -> None:
    _require_columns(df, FRAME_COLUMNS, "frames")
    frames = df["frame"].to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise SchemaError("frames table 'frame' column must be contiguous 0..n-1")


def validate_shots(df: pd.DataFrame) -> None:
    _require_columns(df, SHOT_COLUMNS, "shots")
