import pytest

from badminton_analyzer.video_io import iter_frames, read_video_meta
from tests.helpers import make_video


def test_read_video_meta(tmp_path):
    meta = read_video_meta(make_video(tmp_path / "clip one.mp4", n_frames=30))
    assert (meta.n_frames, meta.width, meta.height) == (30, 160, 90)
    assert meta.fps == pytest.approx(30.0)


def test_iter_frames_yields_bgr_frames(tmp_path):
    frames = list(iter_frames(make_video(tmp_path / "a.mp4", n_frames=12)))
    assert len(frames) == 12 and frames[0].shape == (90, 160, 3)


def test_read_video_meta_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_video_meta(tmp_path / "nope.mp4")
