from pathlib import Path

import pytest

import torch
from torch.utils.data import DataLoader

from badminton_analyzer.tracking import tracknetv3_launcher
from badminton_analyzer.tracking.tracknetv3 import TrackNetV3Tracker


def fake_runner(rows: str):
    def run(cmd, cwd):
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        stem = Path(cmd[cmd.index("--video_file") + 1]).stem
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{stem}_ball.csv").write_text("Frame,Visibility,X,Y\n" + rows)

    return run


def make_tracker(tmp_path, runner):
    return TrackNetV3Tracker(
        tmp_path / "repo", tmp_path / "t.pt", tmp_path / "i.pt", tmp_path / "work dir", runner=runner
    )


def test_tracker_returns_ball_table(tmp_path):
    df = make_tracker(tmp_path, fake_runner("0,1,10,20\n1,0,0,0\n")).track(tmp_path / "my clips" / "rally 1.mp4")
    assert df["visible"].tolist() == [True, False]


def test_command_keeps_paths_with_spaces_intact(tmp_path):
    cmd = make_tracker(tmp_path, fake_runner("")).build_command(Path("C:/a b/rally 1.mp4"), Path("C:/out dir"))
    assert "C:/a b/rally 1.mp4" in [c.replace("\\", "/") for c in cmd]


def test_video_arg_uses_forward_slashes(tmp_path):
    # TrackNetV3's predict.py derives the output name with video_file.split('/').
    cmd = make_tracker(tmp_path, fake_runner("")).build_command(Path("C:/a b/rally 1.mp4"), Path("C:/out dir"))
    video_arg = cmd[cmd.index("--video_file") + 1]
    assert "\\" not in video_arg and video_arg.endswith("/rally 1.mp4")


def test_command_runs_predict_through_launcher(tmp_path):
    cmd = make_tracker(tmp_path, fake_runner("")).build_command(Path("v.mp4"), Path("out"))
    assert Path(cmd[1]) == Path(tracknetv3_launcher.__file__) and cmd[2] == "predict.py"


def test_launcher_forces_single_process_loading(monkeypatch):
    monkeypatch.setattr(DataLoader, "__init__", DataLoader.__init__)  # restored after the test
    tracknetv3_launcher.force_single_process_loading()
    loader = DataLoader(torch.utils.data.TensorDataset(torch.zeros(4, 1)), batch_size=2, num_workers=4)
    assert loader.num_workers == 0


def test_missing_output_raises(tmp_path):
    with pytest.raises(RuntimeError, match="_ball.csv"):
        make_tracker(tmp_path, lambda cmd, cwd: None).track(tmp_path / "x.mp4")
