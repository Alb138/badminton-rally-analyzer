"""NYCU Shuttlecock Trajectory Dataset: shuttle labels and rally index.

Each match folder holds per-frame labels ``<rally>_ball.csv`` (header
``Frame,Visibility,X,Y``) and the rally clips ``<rally>.mp4``. The published
download uses ``ball_trajectory/`` and ``rally_video/`` subfolders, split into
``profession_dataset/`` and ``amateur_dataset/``. TrackNetV3's repo uses
``csv/`` and ``video/``. Both layouts are accepted. TrackNetV3 writes its
predictions in the same CSV format.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LAYOUTS = (("csv", "video"), ("ball_trajectory", "rally_video"))  # (label dir, video dir)


@dataclass(frozen=True)
class NycuRally:
    match_id: str
    rally_id: str
    video_path: Path
    label_path: Path


def read_ball_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    visible = raw["Visibility"].astype(int).to_numpy() == 1
    return pd.DataFrame(
        {
            "frame": raw["Frame"].astype("int64").to_numpy(),
            "visible": visible,
            "x": np.where(visible, raw["X"].astype(float), np.nan),
            "y": np.where(visible, raw["Y"].astype(float), np.nan),
        }
    )


def index_nycu(root: Path) -> list[NycuRally]:
    root = Path(root)
    rallies: list[NycuRally] = []
    unpaired = 0
    for label_dir, video_dir in LAYOUTS:
        for label_path in root.glob(f"**/{label_dir}/*_ball.csv"):
            rally_id = label_path.stem.removesuffix("_ball")
            match_dir = label_path.parent.parent
            video_path = match_dir / video_dir / f"{rally_id}.mp4"
            if not video_path.exists():
                unpaired += 1
                continue
            match_id = match_dir.relative_to(root).as_posix()
            rallies.append(NycuRally(match_id, rally_id, video_path, label_path))
    if unpaired:
        warnings.warn(f"skipped {unpaired} label file(s) with no matching video", UserWarning, stacklevel=2)
    return sorted(rallies, key=lambda r: (r.match_id, r.rally_id))
