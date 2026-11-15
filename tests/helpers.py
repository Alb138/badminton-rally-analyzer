"""Shared test utilities: synthetic videos, frames tables and court clicks."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from badminton_analyzer.schema import empty_frames_table

# Court corner clicks, ordered far_left, far_right, near_right, near_left.
CLICKS_160 = [(40, 20), (120, 20), (150, 85), (10, 85)]
CLICKS_720 = [(400, 200), (880, 200), (1100, 650), (180, 650)]


def make_video(path, n_frames: int = 30, size: tuple[int, int] = (160, 90), fps: float = 30.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for i in range(n_frames):
        writer.write(np.full((height, width, 3), (i * 7) % 256, np.uint8))
    writer.release()
    return path


def build_frames(n_frames: int, shuttle=None, near_bbox=None, far_bbox=None) -> pd.DataFrame:
    df = empty_frames_table(n_frames)
    for frame, (x, y) in (shuttle or {}).items():
        if 0 <= frame < n_frames:
            df.at[frame, "shuttle_visible"] = True
            df.at[frame, "shuttle_x"] = float(x)
            df.at[frame, "shuttle_y"] = float(y)
    for pid, bbox in (("near", near_bbox), ("far", far_bbox)):
        if bbox is not None:
            df[f"{pid}_detected"] = True
            for col, value in zip(("x1", "y1", "x2", "y2"), bbox):
                df[f"{pid}_{col}"] = float(value)
    return df


def polyline(keyframes: list[tuple[int, tuple[float, float]]]) -> dict[int, tuple[float, float]]:
    out = {keyframes[0][0]: tuple(map(float, keyframes[0][1]))}
    for (f0, p0), (f1, p1) in zip(keyframes, keyframes[1:]):
        for f in range(f0, f1 + 1):
            t = (f - f0) / (f1 - f0)
            out[f] = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
    return out
