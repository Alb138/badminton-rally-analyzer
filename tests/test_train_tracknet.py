import json

import pandas as pd
import pytest
import torch

from badminton_analyzer.data.splits import save_splits
from badminton_analyzer.tracking.frames_cache import cache_rally
from badminton_analyzer.tracking.train_tracknet import TrainConfig, train
from tests.helpers import make_video

MATCHES = ["Professional/m1", "Professional/m2", "Professional/m3"]


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    for k, match in enumerate(MATCHES):
        video = make_video(tmp_path / f"v{k}.mp4", n_frames=9, size=(128, 64))
        labels = tmp_path / f"l{k}.csv"
        labels.write_text("Frame,Visibility,X,Y\n" + "".join(f"{i},1,{20 + 8 * i},30\n" for i in range(9)))
        cache_rally(video, labels, root / match / "1_01_00", size=(64, 32))
    return root


def config(cache, splits, out, **kw):
    return TrainConfig(cache, splits, out, epochs=1, batch_size=2, num_workers=0, size=(64, 32), amp=False,
                       device="cpu", **kw)


def test_one_epoch_writes_checkpoint_and_metrics(cache, tmp_path):
    splits = tmp_path / "splits.json"
    save_splits({"train": [MATCHES[0]], "val": [MATCHES[1]], "test": [MATCHES[2]]}, splits)
    best = train(config(cache, splits, tmp_path / "run"))
    assert best.exists() and torch.load(best)["size"] == [64, 32]
    assert len(pd.read_csv(tmp_path / "run" / "metrics.csv")) == 1
    assert json.loads((tmp_path / "run" / "config.json").read_text())["seed"] == 42


def test_overlapping_splits_are_rejected(cache, tmp_path):
    splits = tmp_path / "leaky.json"
    save_splits({"train": [MATCHES[0]], "val": [MATCHES[1]], "test": [MATCHES[0]]}, splits)
    with pytest.raises(ValueError, match="Professional/m1"):
        train(config(cache, splits, tmp_path / "run"))
