"""Shot classification interface, the Phase 1 placeholder, and the shots table."""
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from badminton_analyzer.schema import SHOT_COLUMNS, empty_shots_table


class ShotClassifier(Protocol):
    def classify(self, video_path: Path, frames: pd.DataFrame, hits: pd.DataFrame) -> list[str]:
        """Return one shot label per row of ``hits``."""
        ...


class UnknownShotClassifier:
    """Phase 1 placeholder until the shot-recognition models exist (Phase 3)."""

    def classify(self, video_path: Path, frames: pd.DataFrame, hits: pd.DataFrame) -> list[str]:
        return ["unknown"] * len(hits)


def build_shots_table(frames: pd.DataFrame, hits: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if len(labels) != len(hits):
        raise ValueError(f"got {len(labels)} labels for {len(hits)} hits")
    if hits.empty:
        return empty_shots_table()
    rows = frames.set_index("frame")
    court_x, court_y = [], []
    for frame, hitter in zip(hits["hit_frame"], hits["hitter"]):
        detected = bool(rows.at[frame, f"{hitter}_detected"])
        court_x.append(float(rows.at[frame, f"{hitter}_court_x"]) if detected else np.nan)
        court_y.append(float(rows.at[frame, f"{hitter}_court_y"]) if detected else np.nan)
    return pd.DataFrame(
        {
            "shot_idx": np.arange(len(hits), dtype=np.int64),
            "hit_frame": hits["hit_frame"].to_numpy(dtype=np.int64),
            "hitter": hits["hitter"].astype(object).to_numpy(),
            "shot_type": np.array(labels, dtype=object),
            "hitter_court_x": np.array(court_x, dtype=float),
            "hitter_court_y": np.array(court_y, dtype=float),
        },
        columns=SHOT_COLUMNS,
    )
