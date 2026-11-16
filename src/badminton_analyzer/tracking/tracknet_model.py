"""TrackNet-style shuttle heatmap model (after TrackNetV2, Sun et al. 2020), implemented for this project.

Input: 3 consecutive RGB frames stacked on the channel axis. Output: one heatmap per frame.
A VGG-style encoder-decoder (U-Net) with skip connections keeps the tiny, blurred shuttle
localizable, and seeing 3 frames at once lets it use motion when a single frame is ambiguous.
"""
import cv2
import numpy as np
import torch
from torch import nn


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _block(cin: int, cout: int, n: int) -> nn.Sequential:
    return nn.Sequential(_conv(cin, cout), *[_conv(cout, cout) for _ in range(n - 1)])


class TrackNetV2(nn.Module):
    """(B, 3*in_frames, H, W) -> (B, out_frames, H, W) heatmaps in [0, 1]; H and W divisible by 8."""

    def __init__(self, in_frames: int = 3, out_frames: int = 3):
        super().__init__()
        self.enc1 = _block(3 * in_frames, 64, 2)
        self.enc2 = _block(64, 128, 2)
        self.enc3 = _block(128, 256, 3)
        self.bottleneck = _block(256, 512, 3)
        self.dec3 = _block(512 + 256, 256, 3)
        self.dec2 = _block(256 + 128, 128, 2)
        self.dec1 = _block(128 + 64, 64, 2)
        self.head = nn.Conv2d(64, out_frames, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


def wbce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TrackNetV2's weighted BCE: confident mistakes cost more, which suits a mostly-empty heatmap."""
    p = pred.clamp(1e-7, 1 - 1e-7)
    return -((1 - p) ** 2 * target * torch.log(p) + p**2 * (1 - target) * torch.log(1 - p)).mean()


def decode_heatmap(hm: np.ndarray, threshold: float = 0.5) -> tuple[bool, float, float]:
    """Centroid (x, y) of the largest blob above ``threshold``, or (False, nan, nan)."""
    binary = (np.asarray(hm) > threshold).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return False, float("nan"), float("nan")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[largest]
    return True, float(cx), float(cy)
