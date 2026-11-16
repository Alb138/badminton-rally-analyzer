import numpy as np
import pytest

from badminton_analyzer.tracking.factory import make_tracker
from badminton_analyzer.tracking.yolo_tracker import YoloShuttleTracker
from tests.helpers import make_video


def test_highest_confidence_box_wins(tmp_path):
    boxes = {0: np.array([[10, 10, 0.3], [50, 20, 0.9]]), 1: np.zeros((0, 3))}
    calls = iter(range(2))
    t = YoloShuttleTracker(None, predict_fn=lambda frame: boxes[next(calls)])
    df = t.track(make_video(tmp_path / "v.mp4", n_frames=2))
    assert df["visible"].tolist() == [True, False] and (df.loc[0, "x"], df.loc[0, "y"]) == (50, 20)
    assert np.isnan(df.loc[1, "x"])


def test_factory_requires_existing_yolo_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        make_tracker({"kind": "yolo", "weights": str(tmp_path / "nope.pt")})
