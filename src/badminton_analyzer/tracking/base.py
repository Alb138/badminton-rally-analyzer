"""Interface every shuttle tracker implements."""
from pathlib import Path
from typing import Protocol

import pandas as pd


class ShuttleTracker(Protocol):
    def track(self, video_path: Path) -> pd.DataFrame:
        """Return one row per frame with columns frame, visible, x, y (NaN when not visible)."""
        ...
