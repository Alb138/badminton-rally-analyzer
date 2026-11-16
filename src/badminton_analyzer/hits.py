"""Rule-based hit detection from the shuttle track and player boxes.

A hit is where the shuttle changes direction sharply next to a player, plus the
serve, which is where the shuttle first starts moving fast next to a player. All
thresholds are in seconds or fractions of the frame height, so they behave the
same at any fps or resolution.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from badminton_analyzer.schema import PLAYER_IDS

HIT_COLUMNS = ["hit_frame", "hitter"]
SERVE_SCORE = 180.0


@dataclass(frozen=True)
class HitParams:
    min_gap_s: float = 0.25
    min_turn_deg: float = 45.0
    max_hitter_dist_frac: float = 0.12
    smooth_window: int = 3
    max_gap_s: float = 0.1
    serve_speed_frac: float = 0.2  # frame heights per second
    bbox_top_extend: float = 0.3
    merge_same_hitter_s: float = 0.6  # same-hitter hits further apart are kept (opponent's hit was missed)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"hit_frame": pd.Series(dtype="int64"), "hitter": pd.Series(dtype="object")})


def _candidates(frames: pd.DataFrame, fps: float, frame_height: int, p: HitParams) -> list[tuple[int, float, float, float]]:
    """Return (frame, x, y, score) for every turn and the serve."""
    vis = frames[frames["shuttle_visible"]]
    if vis.empty:
        return []
    f = vis["frame"].to_numpy()
    xy = vis[["shuttle_x", "shuttle_y"]].to_numpy(dtype=float)
    max_gap = max(1, round(p.max_gap_s * fps))
    segments = np.split(np.arange(len(f)), np.where(np.diff(f) > max_gap)[0] + 1)
    serve_speed = p.serve_speed_frac * frame_height  # px per second
    min_step = 0.25 * serve_speed / fps  # ignore turns of a near-stationary shuttle

    out = []
    for k, idx in enumerate(segments):
        sf, sxy = f[idx], xy[idx]
        if k == 0 and len(idx) >= 2:
            speeds = np.linalg.norm(np.diff(sxy, axis=0), axis=1) / np.diff(sf) * fps
            fast = np.flatnonzero(speeds > serve_speed)
            if fast.size:
                i = fast[0]
                out.append((int(sf[i]), *sxy[i], SERVE_SCORE))
        if len(idx) < 3:
            continue
        raw_steps = np.linalg.norm(np.diff(sxy, axis=0), axis=1) / np.diff(sf)
        smooth = pd.DataFrame(sxy).rolling(p.smooth_window, center=True, min_periods=1).mean().to_numpy()
        v = np.diff(smooth, axis=0)
        norms = np.linalg.norm(v, axis=1)
        for i in range(1, len(smooth) - 1):
            if raw_steps[i - 1] < min_step or raw_steps[i] < min_step or norms[i - 1] == 0 or norms[i] == 0:
                continue
            cos = np.dot(v[i - 1], v[i]) / (norms[i - 1] * norms[i])
            angle = math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))
            if angle >= p.min_turn_deg:
                out.append((int(sf[i]), *sxy[i], angle))
    return out


def _nearest_player(row: pd.Series, x: float, y: float, p: HitParams) -> tuple[str, float] | None:
    best = None
    for pid in PLAYER_IDS:
        if not row[f"{pid}_detected"]:
            continue
        x1, y1, x2, y2 = (row[f"{pid}_{c}"] for c in ("x1", "y1", "x2", "y2"))
        top = y1 - p.bbox_top_extend * (y2 - y1)  # room for a raised racket
        dist = math.hypot(max(x1 - x, 0.0, x - x2), max(top - y, 0.0, y - y2))
        if best is None or dist < best[1]:
            best = (pid, dist)
    return best


def detect_hits(frames: pd.DataFrame, fps: float, frame_height: int, params: HitParams) -> pd.DataFrame:
    rows = frames.set_index("frame")
    max_dist = params.max_hitter_dist_frac * frame_height
    hits = []
    for frame, x, y, score in sorted(_candidates(frames, fps, frame_height, params)):
        nearest = _nearest_player(rows.loc[frame], x, y, params)
        if nearest is not None and nearest[1] <= max_dist:
            hits.append((frame, nearest[0], score))

    min_gap = max(1, round(params.min_gap_s * fps))
    clustered: list[tuple[int, str, float]] = []
    last_frame = None
    for hit in hits:
        if clustered and hit[0] - last_frame < min_gap:
            if hit[2] > clustered[-1][2]:
                clustered[-1] = hit
        else:
            clustered.append(hit)
        last_frame = hit[0]

    # Singles: hitters alternate. Merge only same-hitter hits close in time; further apart, the
    # opponent's hit in between was probably missed, and merging would cascade one error over the rally.
    merge_gap = params.merge_same_hitter_s * fps
    kept: list[tuple[int, str, float]] = []
    for hit in clustered:
        if kept and kept[-1][1] == hit[1] and hit[0] - kept[-1][0] <= merge_gap:
            if hit[2] > kept[-1][2]:
                kept[-1] = hit
        else:
            kept.append(hit)

    if not kept:
        return _empty()
    return pd.DataFrame(
        {"hit_frame": np.array([h[0] for h in kept], dtype=np.int64), "hitter": [h[1] for h in kept]}
    )
