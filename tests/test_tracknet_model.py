import numpy as np
import pytest
import torch

from badminton_analyzer.tracking.tracknet_model import TrackNetV2, decode_heatmap, wbce_loss


def test_forward_shape_and_range():
    m = TrackNetV2().eval()
    with torch.no_grad():
        out = m(torch.rand(2, 9, 32, 64))
    assert out.shape == (2, 3, 32, 64) and 0 <= out.min() and out.max() <= 1


def test_parameter_count_is_tracknetv2_sized():
    n = sum(p.numel() for p in TrackNetV2().parameters())
    assert 5e6 < n < 20e6


def test_loss_prefers_the_right_heatmap():
    target = torch.zeros(1, 3, 32, 64)
    target[0, :, 10, 20] = 1.0
    good = target.clamp(0.01, 0.99)
    bad = (1 - target).clamp(0.01, 0.99)
    assert wbce_loss(good, target) < wbce_loss(bad, target) and torch.isfinite(wbce_loss(good, target))


def test_decode_returns_centroid_of_largest_blob():
    hm = np.zeros((32, 64), np.float32)
    hm[10:13, 20:23] = 0.9
    hm[1, 1] = 0.9
    assert decode_heatmap(hm) == (True, pytest.approx(21.0), pytest.approx(11.0))


def test_decode_empty_heatmap_is_invisible():
    visible, x, y = decode_heatmap(np.full((32, 64), 0.2, np.float32))
    assert not visible and np.isnan(x) and np.isnan(y)
