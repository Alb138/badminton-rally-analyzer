"""Rally statistics from the frames and shots tables, as a JSON-safe dict."""
import math

import numpy as np
import pandas as pd

from badminton_analyzer.schema import PLAYER_IDS

MEDIAN_WINDOW = 5


def _num(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _distance_m(frames: pd.DataFrame, pid: str, max_step_m: float) -> float:
    det = frames[frames[f"{pid}_detected"]]
    if len(det) < 2:
        return 0.0
    raw = det[[f"{pid}_court_x", f"{pid}_court_y"]]
    # Median-filter only full windows; partial windows at the ends would shrink real movement.
    xy = raw.rolling(MEDIAN_WINDOW, center=True).median().fillna(raw)
    steps = np.linalg.norm(np.diff(xy.to_numpy(), axis=0), axis=1)
    consecutive = np.diff(det["frame"].to_numpy()) == 1
    keep = consecutive & np.isfinite(steps) & (steps <= max_step_m)
    return float(steps[keep].sum())


def rally_stats(frames: pd.DataFrame, shots: pd.DataFrame, fps: float, max_step_m: float = 0.5) -> dict:
    n_frames = len(frames)
    return {
        "n_frames": n_frames,
        "fps": float(fps),
        "duration_s": round(n_frames / fps, 3) if fps else None,
        "n_shots": len(shots),
        "shot_counts": {str(k): int(v) for k, v in shots["shot_type"].value_counts().items()},
        "shots": [
            {
                "shot_idx": int(s.shot_idx),
                "hit_frame": int(s.hit_frame),
                "time_s": round(s.hit_frame / fps, 3),
                "hitter": str(s.hitter),
                "shot_type": str(s.shot_type),
                "court_x": _num(s.hitter_court_x),
                "court_y": _num(s.hitter_court_y),
            }
            for s in shots.itertuples(index=False)
        ],
        "players": {
            pid: {
                "distance_m": round(_distance_m(frames, pid, max_step_m), 3),
                "detected_frac": round(float(frames[f"{pid}_detected"].mean()) if n_frames else 0.0, 3),
            }
            for pid in PLAYER_IDS
        },
    }
