"""Rally statistics from the frames and shots tables, as a JSON-safe dict."""
import math

import numpy as np
import pandas as pd

from badminton_analyzer.schema import PLAYER_IDS

# Distance covered is measured on a fixed time grid so foot-position noise doesn't add up
# faster at higher frame rates: smooth over SMOOTH_S, then sum steps every SAMPLE_S.
SMOOTH_S = 0.2
SAMPLE_S = 0.2


def _num(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _distance_m(frames: pd.DataFrame, pid: str, fps: float, max_speed_mps: float) -> float:
    det = frames[frames[f"{pid}_detected"]]
    if len(det) < 2 or not fps:
        return 0.0
    raw = det[[f"{pid}_court_x", f"{pid}_court_y"]]
    window = max(1, round(SMOOTH_S * fps))
    # Median-filter only full windows; partial windows at the ends would shrink real movement.
    xy = raw.rolling(window, center=True).median().fillna(raw).to_numpy()
    frame_no = det["frame"].to_numpy()
    k = max(1, round(SAMPLE_S * fps))
    max_step = max_speed_mps * k / fps  # anything longer is a detection jump, not movement
    total, last = 0.0, 0
    for i in range(1, len(frame_no)):
        gap = frame_no[i] - frame_no[last]
        if gap < k:
            continue
        if gap == k:  # no missed frames in between
            step = float(np.linalg.norm(xy[i] - xy[last]))
            if np.isfinite(step) and step <= max_step:
                total += step
        last = i
    return total


def rally_stats(frames: pd.DataFrame, shots: pd.DataFrame, fps: float, max_speed_mps: float = 10.0) -> dict:
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
                "distance_m": round(_distance_m(frames, pid, fps, max_speed_mps), 3),
                "detected_frac": round(float(frames[f"{pid}_detected"].mean()) if n_frames else 0.0, 3),
            }
            for pid in PLAYER_IDS
        },
    }
