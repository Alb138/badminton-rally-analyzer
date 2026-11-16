"""Training windows for the TrackNet-style model: 3 consecutive frames in, 3 heatmaps out."""
import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from badminton_analyzer.data.nycu import read_ball_csv

SEQ_LEN = 3
SIGMA = 2.5  # px, in input space


def gaussian_heatmap(x: float, y: float, visible: bool, size=(512, 288), sigma: float = SIGMA) -> np.ndarray:
    """(H, W) float32 Gaussian with peak 1.0 at (x, y); all zeros when not visible."""
    w, h = size
    if not visible or not (np.isfinite(x) and np.isfinite(y)):
        return np.zeros((h, w), np.float32)
    x, y = float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    return np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2)).astype(np.float32)


class RallyWindows(Dataset):
    """Windows of SEQ_LEN frames from cached rallies (see frames_cache.cache_rally).

    Items are numpy arrays (the default DataLoader collate turns them into tensors):
    x (3*SEQ_LEN, H, W) in [0, 1], y (SEQ_LEN, H, W) heatmaps, coords (SEQ_LEN, 3) as
    [visible, x, y] in input pixels (0, 0 when not visible).
    """

    def __init__(self, rally_dirs: list[Path], size=(512, 288), stride: int = 1, augment: bool = False,
                 flip_p: float = 0.5, seed: int = 42):
        self.size = tuple(size)
        self.augment, self.flip_p, self.seed = augment, flip_p, seed
        self.rallies: list[tuple[Path, np.ndarray]] = []
        self.items: list[tuple[int, int]] = []
        for k, rally_dir in enumerate(map(Path, rally_dirs)):
            meta = json.loads((rally_dir / "meta.json").read_text(encoding="utf-8"))
            n = meta["n_frames"]
            labels = read_ball_csv(rally_dir / "labels.csv").set_index("frame").reindex(range(n))
            visible = labels["visible"].eq(True).to_numpy()
            sx, sy = self.size[0] / meta["width"], self.size[1] / meta["height"]
            coords = np.zeros((n, 3), np.float32)
            coords[visible, 0] = 1.0
            coords[visible, 1] = labels["x"].to_numpy()[visible] * sx
            coords[visible, 2] = labels["y"].to_numpy()[visible] * sy
            self.rallies.append((rally_dir, coords))
            self.items += [(k, start) for start in range(0, n - SEQ_LEN + 1, stride)]

    def __len__(self) -> int:
        return len(self.items)

    def _frame(self, rally_dir: Path, index: int) -> np.ndarray:
        img = cv2.imread(str(rally_dir / f"{index:05d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"missing cached frame {index} in {rally_dir}")
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        k, start = self.items[idx]
        rally_dir, all_coords = self.rallies[k]
        frames = np.stack([self._frame(rally_dir, start + i) for i in range(SEQ_LEN)])  # (S, H, W, 3)
        coords = all_coords[start : start + SEQ_LEN].copy()
        if self.augment and np.random.default_rng(self.seed + idx).random() < self.flip_p:
            frames = frames[:, :, ::-1]
            visible = coords[:, 0] > 0
            coords[visible, 1] = self.size[0] - 1 - coords[visible, 1]
        x = np.ascontiguousarray(frames.transpose(0, 3, 1, 2).reshape(-1, self.size[1], self.size[0]),
                                 dtype=np.float32) / 255.0
        y = np.stack([gaussian_heatmap(cx, cy, bool(v), self.size) for v, cx, cy in coords])
        return x, y, coords
