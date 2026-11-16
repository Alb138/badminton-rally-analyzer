import pytest

from badminton_analyzer.tracking.factory import make_tracker
from badminton_analyzer.tracking.tracknetv3 import TrackNetV3Tracker


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        make_tracker({"kind": "nope"})


def test_tracknetv3_kind_builds_wrapper(tmp_path):
    t = make_tracker({"kind": "tracknetv3", "repo_dir": str(tmp_path), "tracknet_ckpt": "a.pt",
                      "inpaintnet_ckpt": "b.pt", "work_dir": str(tmp_path / "w"), "batch_size": 2})
    assert isinstance(t, TrackNetV3Tracker) and t.batch_size == 2
