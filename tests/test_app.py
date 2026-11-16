from pathlib import Path

import numpy as np

from badminton_analyzer import app
from badminton_analyzer.app import add_corner, analyze, draw_corner_markers, frame_at
from badminton_analyzer.shots import UnknownShotClassifier
from tests.helpers import CLICKS_160, CONFIG, FakePose, FakeTracker, make_video


def test_add_corner_caps_at_four():
    pts = []
    for xy in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]:
        pts = add_corner(pts, xy)
    assert pts == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_draw_corner_markers_copies_input():
    img = np.zeros((90, 160, 3), np.uint8)
    out = draw_corner_markers(img, [(10, 10)])
    assert out.any() and not img.any()


def test_frame_at_returns_requested_frame(tmp_path):
    # Broadcast clips often open on a replay camera, so calibration must not be tied to frame 0.
    video = str(make_video(tmp_path / "v.mp4", n_frames=12))
    assert abs(int(frame_at(video, 10)[45, 80, 0]) - 70) <= 4  # make_video paints frame i with grey (i * 7)
    assert frame_at(video, 99).shape == (90, 160, 3)  # past the end -> last frame


def test_analyze_needs_four_corners(tmp_path):
    assert analyze(str(make_video(tmp_path / "a.mp4")), [(1, 1)], str(CONFIG), tmp_path) == (
        None, None, "Click all 4 court corners first (1/4)."
    )


def test_analyze_runs_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "get_components", lambda cfg: (FakeTracker(), FakePose(), UnknownShotClassifier()))
    video_out, stats, status = analyze(str(make_video(tmp_path / "a b.mp4", n_frames=60)), CLICKS_160, str(CONFIG), tmp_path)
    assert Path(video_out).exists() and stats["n_frames"] == 60 and status.startswith("Done")
