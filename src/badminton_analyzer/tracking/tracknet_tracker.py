"""Shuttle tracker backed by the project's own TrackNet-style model."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from badminton_analyzer.tracking.tracknet_data import SEQ_LEN
from badminton_analyzer.tracking.tracknet_model import TrackNetV2, decode_heatmap
from badminton_analyzer.video_io import iter_frames


class TrackNetTracker:
    """Runs non-overlapping 3-frame windows; the last window repeats the final frame as padding."""

    def __init__(self, weights: Path | None, device: str | None = None, batch_size: int = 4, threshold: float = 0.5,
                 model: nn.Module | None = None, size: tuple[int, int] | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            if weights is None:
                raise ValueError("TrackNetTracker needs weights or an injected model")
            checkpoint = torch.load(Path(weights), map_location="cpu")
            model = TrackNetV2()
            model.load_state_dict(checkpoint["model"])
            size = tuple(checkpoint["size"])
        elif size is None:
            raise ValueError("an injected model needs its input size")
        self.model = model.to(self.device).eval()
        self.size = tuple(size)
        self.batch_size = batch_size
        self.threshold = threshold

    @torch.no_grad()
    def track(self, video_path: Path) -> pd.DataFrame:
        w, h = self.size
        frames, src_w, src_h = [], None, None
        for frame in iter_frames(Path(video_path)):
            src_h, src_w = frame.shape[:2]
            frames.append(cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        n = len(frames)
        if n == 0:
            return pd.DataFrame({"frame": pd.Series(dtype="int64"), "visible": pd.Series(dtype=bool),
                                 "x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
        frames += [frames[-1]] * ((-n) % SEQ_LEN)
        windows = np.stack(frames).reshape(-1, SEQ_LEN, h, w, 3)  # (N, S, H, W, 3)
        heatmaps = []
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size].transpose(0, 1, 4, 2, 3).reshape(-1, SEQ_LEN * 3, h, w)
            x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32) / 255.0).to(self.device)
            heatmaps.append(self.model(x).float().cpu().numpy().reshape(-1, h, w))
        heatmaps = np.concatenate(heatmaps)[:n]
        rows = []
        for i, hm in enumerate(heatmaps):
            visible, px, py = decode_heatmap(hm, self.threshold)
            rows.append((i, visible, px * src_w / w if visible else np.nan, py * src_h / h if visible else np.nan))
        return pd.DataFrame(rows, columns=["frame", "visible", "x", "y"])
