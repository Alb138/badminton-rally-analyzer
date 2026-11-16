"""Annotated video: shuttle trail, player skeletons, shot labels and a top-down court inset."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from badminton_analyzer.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    SHORT_SERVICE_FROM_NET_M,
    SINGLES_WIDTH_M,
)
from badminton_analyzer.schema import PLAYER_IDS
from badminton_analyzer.video_io import iter_frames, read_video_meta

# BGR colours
PLAYER_COLORS = {"near": (0, 165, 255), "far": (255, 200, 0)}
SHUTTLE_COLOR = (0, 255, 255)
COURT_BG = (60, 110, 50)
COURT_LINE = (235, 235, 235)
LONG_SERVICE_FROM_BASELINE_M = 0.76
INSET_MARGIN = 8
KPT_MIN_CONF = 0.3
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


@dataclass(frozen=True)
class RenderParams:
    trail_length: int = 8
    label_duration_s: float = 0.7
    inset_width_frac: float = 0.18


def inset_size(frame_w: int, frame_h: int, width_frac: float) -> tuple[int, int]:
    width = max(1, round(frame_w * width_frac))
    height = round(width * COURT_LENGTH_M / COURT_WIDTH_M)
    if height > 0.9 * frame_h:
        height = max(1, int(0.9 * frame_h))
        width = max(1, round(height * COURT_WIDTH_M / COURT_LENGTH_M))
    return width, height


def draw_court_inset(width_px: int, players: dict[str, tuple[float, float] | None]) -> np.ndarray:
    height = round(width_px * COURT_LENGTH_M / COURT_WIDTH_M)
    img = np.full((height, width_px, 3), COURT_BG, np.uint8)
    sx, sy = (width_px - 1) / COURT_WIDTH_M, (height - 1) / COURT_LENGTH_M

    def pt(x: float, y: float) -> tuple[int, int]:
        return int(round(x * sx)), int(round(y * sy))

    def line(x0, y0, x1, y1, thickness=1):
        cv2.line(img, pt(x0, y0), pt(x1, y1), COURT_LINE, thickness)

    side = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2
    cv2.rectangle(img, pt(0, 0), pt(COURT_WIDTH_M, COURT_LENGTH_M), COURT_LINE, 1)
    for x in (side, COURT_WIDTH_M - side):
        line(x, 0, x, COURT_LENGTH_M)
    for y in (NET_Y_M - SHORT_SERVICE_FROM_NET_M, NET_Y_M + SHORT_SERVICE_FROM_NET_M,
              LONG_SERVICE_FROM_BASELINE_M, COURT_LENGTH_M - LONG_SERVICE_FROM_BASELINE_M):
        line(0, y, COURT_WIDTH_M, y)
    line(COURT_WIDTH_M / 2, 0, COURT_WIDTH_M / 2, NET_Y_M - SHORT_SERVICE_FROM_NET_M)
    line(COURT_WIDTH_M / 2, NET_Y_M + SHORT_SERVICE_FROM_NET_M, COURT_WIDTH_M / 2, COURT_LENGTH_M)
    line(0, NET_Y_M, COURT_WIDTH_M, NET_Y_M, thickness=2)

    radius = max(2, width_px // 20)
    for pid, pos in players.items():
        if pos is None or not np.isfinite(pos).all():
            continue
        x = min(max(pos[0], 0.0), COURT_WIDTH_M)
        y = min(max(pos[1], 0.0), COURT_LENGTH_M)
        cv2.circle(img, pt(x, y), radius, PLAYER_COLORS[pid], -1)
    return img


def _draw_skeleton(img: np.ndarray, kpts: np.ndarray, color: tuple[int, int, int]) -> None:
    ok = np.isfinite(kpts).all(axis=1) & (kpts[:, 2] >= KPT_MIN_CONF)
    for a, b in COCO_SKELETON:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(int(v) for v in kpts[a, :2]), tuple(int(v) for v in kpts[b, :2]), color, 2)
    for x, y, _ in kpts[ok]:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)


def draw_frame(
    frame: np.ndarray, frame_idx: int, frames: pd.DataFrame, shots: pd.DataFrame, fps: float, params: RenderParams
) -> np.ndarray:
    out = frame.copy()
    if frame_idx >= len(frames):
        return out
    row = frames.iloc[frame_idx]
    h, w = out.shape[:2]

    for pid in PLAYER_IDS:
        if row[f"{pid}_detected"]:
            _draw_skeleton(out, np.asarray(row[f"{pid}_kpts"], dtype=float).reshape(17, 3), PLAYER_COLORS[pid])

    start = max(0, frame_idx - params.trail_length + 1)
    trail = frames.iloc[start : frame_idx + 1]
    vis, xs, ys = trail["shuttle_visible"].to_numpy(), trail["shuttle_x"].to_numpy(), trail["shuttle_y"].to_numpy()
    for i in range(1, len(trail)):
        if vis[i - 1] and vis[i]:
            cv2.line(out, (int(xs[i - 1]), int(ys[i - 1])), (int(xs[i]), int(ys[i])), SHUTTLE_COLOR, 2)
    if row["shuttle_visible"]:
        cv2.circle(out, (int(row["shuttle_x"]), int(row["shuttle_y"])), 3, SHUTTLE_COLOR, -1)

    label_frames = params.label_duration_s * fps
    active = shots[(shots["hit_frame"] <= frame_idx) & (frame_idx < shots["hit_frame"] + label_frames)]
    for shot in active.itertuples(index=False):
        if row[f"{shot.hitter}_detected"]:
            x, y = int(row[f"{shot.hitter}_x1"]), int(row[f"{shot.hitter}_y1"]) - 8
            cv2.putText(out, str(shot.shot_type), (x, max(y, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        PLAYER_COLORS[shot.hitter], 2, cv2.LINE_AA)

    inset_w, inset_h = inset_size(w, h, params.inset_width_frac)
    players = {
        pid: (row[f"{pid}_court_x"], row[f"{pid}_court_y"]) if row[f"{pid}_detected"] else None for pid in PLAYER_IDS
    }
    inset = cv2.resize(draw_court_inset(inset_w, players), (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    x0, y0 = w - inset_w - INSET_MARGIN, INSET_MARGIN
    if x0 >= 0 and y0 + inset_h <= h:
        out[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
    return out


def render_video(
    video_path: Path, frames: pd.DataFrame, shots: pd.DataFrame, out_path: Path, params: RenderParams
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = read_video_meta(video_path).fps or 30.0
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error"
    )
    try:
        for i, frame in enumerate(iter_frames(video_path)):
            annotated = draw_frame(frame, i, frames, shots, fps, params)
            h, w = annotated.shape[:2]
            if h % 2 or w % 2:  # yuv420p needs even dimensions
                annotated = cv2.copyMakeBorder(annotated, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)
            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    return out_path
