import numpy as np
import pytest

from badminton_analyzer.tracking.frames_cache import cache_rally
from badminton_analyzer.tracking.tracknet_data import RallyWindows, gaussian_heatmap
from tests.helpers import make_video


def test_heatmap_peaks_at_the_point():
    hm = gaussian_heatmap(100.0, 50.0, True)
    assert hm.shape == (288, 512) and hm.max() == pytest.approx(1.0)
    assert np.unravel_index(hm.argmax(), hm.shape) == (50, 100)


def test_invisible_heatmap_is_empty_even_with_zero_coords():
    assert not gaussian_heatmap(0.0, 0.0, False).any()


def test_out_of_frame_point_is_clipped():
    hm = gaussian_heatmap(600.0, -5.0, True)
    assert np.unravel_index(hm.argmax(), hm.shape) == (0, 511)


def rally_cache(tmp_path):  # 7 frames at 128x64; shuttle at source (40, 20), invisible on frame 3
    video = make_video(tmp_path / "v.mp4", n_frames=7, size=(128, 64))
    labels = tmp_path / "1_01_00_ball.csv"
    labels.write_text("Frame,Visibility,X,Y\n" + "".join(
        f"{i},{0 if i == 3 else 1},{0 if i == 3 else 40},{0 if i == 3 else 20}\n" for i in range(7)))
    return cache_rally(video, labels, tmp_path / "c", size=(64, 32)), tmp_path / "c"


def test_cache_and_windows(tmp_path):
    n, cache = rally_cache(tmp_path)
    ds = RallyWindows([cache], size=(64, 32))
    x, y, coords = ds[0]
    assert n == 7 and len(ds) == 5 and x.shape == (9, 32, 64) and y.shape == (3, 32, 64)
    assert coords[0].tolist() == [1.0, 20.0, 10.0]  # (40, 20) scaled by 64/128
    assert not ds[1][1][2].any()  # window 1 = frames 1-3; frame 3 invisible -> empty map


def test_flip_mirrors_image_and_label(tmp_path):
    _, cache = rally_cache(tmp_path)
    _, y, coords = RallyWindows([cache], size=(64, 32), augment=True, flip_p=1.0)[0]
    assert coords[0, 1] == pytest.approx(63 - 20.0)
    assert np.unravel_index(int(y[0].argmax()), y[0].shape)[1] == 43


def test_cache_is_reused_when_complete(tmp_path):
    n, cache = rally_cache(tmp_path)
    stamp = (cache / "00000.jpg").stat().st_mtime_ns
    assert cache_rally(tmp_path / "v.mp4", tmp_path / "1_01_00_ball.csv", cache, size=(64, 32)) == n
    assert (cache / "00000.jpg").stat().st_mtime_ns == stamp
