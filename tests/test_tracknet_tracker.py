import pytest
import torch
from torch import nn

from badminton_analyzer.tracking.factory import make_tracker
from badminton_analyzer.tracking.tracknet_model import TrackNetV2
from badminton_analyzer.tracking.tracknet_tracker import TrackNetTracker
from tests.helpers import make_video


class BlobModel(nn.Module):  # a blob centred at input (x=32, y=16) for every frame
    def forward(self, x):
        out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        out[:, :, 15:18, 31:34] = 0.9
        return out


def test_one_row_per_frame_for_lengths_not_divisible_by_three(tmp_path):
    for n in (1, 2, 7):
        video = make_video(tmp_path / f"{n}.mp4", n_frames=n, size=(128, 64))
        df = TrackNetTracker(None, model=BlobModel(), size=(64, 32)).track(video)
        assert df["frame"].tolist() == list(range(n))


def test_coordinates_are_scaled_to_source_size(tmp_path):
    video = make_video(tmp_path / "v.mp4", n_frames=3, size=(128, 64))
    df = TrackNetTracker(None, model=BlobModel(), size=(64, 32)).track(video)
    assert df["visible"].all() and df["x"].iloc[0] == pytest.approx(64.0) and df["y"].iloc[0] == pytest.approx(32.0)


def test_loads_model_and_size_from_checkpoint(tmp_path):
    ckpt = tmp_path / "best.pt"
    torch.save({"model": TrackNetV2().state_dict(), "epoch": 1, "val_f1": 0.0, "size": [64, 32]}, ckpt)
    df = TrackNetTracker(ckpt, device="cpu").track(make_video(tmp_path / "v.mp4", n_frames=4, size=(128, 64)))
    assert len(df) == 4 and list(df.columns) == ["frame", "visible", "x", "y"]


def test_factory_requires_existing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        make_tracker({"kind": "tracknet", "weights": str(tmp_path / "missing.pt")})
