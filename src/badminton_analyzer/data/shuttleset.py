"""ShuttleSet stroke annotations and the merged shot-class mapping."""
from pathlib import Path

import pandas as pd
import yaml

from badminton_analyzer.schema import SchemaError

REQUIRED_STROKE_COLUMNS = ("rally", "ball_round", "frame_num", "player", "type")
MERGED_CLASSES = ("serve", "net", "smash", "lob", "clear", "drive", "drop", "push_rush")
EXCLUDE = "_exclude"


def load_shuttleset_strokes(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(root).glob("**/set*.csv")):
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_STROKE_COLUMNS if c not in df.columns]
        if missing:
            raise SchemaError(f"{path} is missing stroke columns: {', '.join(missing)}")
        df["match_id"] = path.parent.name
        df["set_id"] = path.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no set*.csv files under {root}")
    return pd.concat(frames, ignore_index=True)


def load_shot_class_map(path: Path) -> dict[str, str]:
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = set(MERGED_CLASSES) | {EXCLUDE}
    bad = sorted({str(v) for v in mapping.values() if v not in allowed})
    if bad:
        raise ValueError(f"unknown merged classes in {path}: {', '.join(bad)}")
    return {str(k): str(v) for k, v in mapping.items()}


def apply_class_map(strokes: pd.DataFrame, class_map: dict[str, str]) -> pd.DataFrame:
    unmapped = sorted(set(strokes["type"].astype(str)) - set(class_map))
    if unmapped:
        raise ValueError(f"raw shot types missing from class map: {', '.join(unmapped)}")
    out = strokes.copy()
    out["shot_class"] = out["type"].astype(str).map(class_map)
    return out[out["shot_class"] != EXCLUDE].reset_index(drop=True)
