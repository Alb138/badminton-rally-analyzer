import numpy as np
import pytest

from badminton_analyzer.render import RenderParams, draw_court_inset, draw_frame, inset_size, render_video
from badminton_analyzer.schema import empty_shots_table
from badminton_analyzer.video_io import read_video_meta
from tests.helpers import build_frames, make_video


def test_inset_shape_follows_court_aspect():
    assert draw_court_inset(100, {"near": (3.05, 10.0), "far": None}).shape == (round(100 * 13.40 / 6.10), 100, 3)


def test_inset_fits_short_wide_frames():
    w, h = inset_size(400, 90, 0.18)
    assert h <= 81 and w / h == pytest.approx(6.10 / 13.40, rel=0.05)


def test_draw_frame_marks_shuttle_without_touching_input():
    frame = np.zeros((90, 160, 3), np.uint8)
    out = draw_frame(frame, 9, build_frames(10, {i: (60, 45) for i in range(10)}), empty_shots_table(), 30.0, RenderParams())
    assert out.shape == frame.shape and out[45, 60].any() and not frame.any()


def test_render_video_keeps_frame_count_with_odd_size(tmp_path):
    src = make_video(tmp_path / "in clip.mp4", n_frames=20, size=(161, 91))
    out = render_video(src, build_frames(20), empty_shots_table(), tmp_path / "out dir" / "annotated.mp4", RenderParams())
    assert read_video_meta(out).n_frames == 20
