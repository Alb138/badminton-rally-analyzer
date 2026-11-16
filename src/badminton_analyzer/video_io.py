"""Reading video metadata and frames with OpenCV."""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    n_frames: int
    width: int
    height: int


def _open(path: Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {path}")
    return cap


def read_video_meta(path: Path) -> VideoMeta:
    cap = _open(path)
    try:
        return VideoMeta(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield frames in BGR order."""
    cap = _open(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()
