from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from badminton_analyzer.tracking import tracknetv3_launcher
from badminton_analyzer.tracking.tracknetv3 import TrackNetV3Tracker
from badminton_analyzer.video_io import read_video_meta
from tests.helpers import make_video


def fake_runner(rows: str, seen: list | None = None):
    def run(cmd, cwd):
        save_dir = Path(cmd[cmd.index("--save_dir") + 1])
        video = Path(cmd[cmd.index("--video_file") + 1])
        if seen is not None:
            seen.append((video, read_video_meta(video)))
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{video.stem}_ball.csv").write_text("Frame,Visibility,X,Y\n" + rows)

    return run


def make_tracker(tmp_path, runner):
    return TrackNetV3Tracker(
        tmp_path / "repo", tmp_path / "t.pt", tmp_path / "i.pt", tmp_path / "work dir", runner=runner
    )


def test_tracker_returns_ball_table(tmp_path):
    video = make_video(tmp_path / "my clips" / "rally 1.mp4", n_frames=5)
    df = make_tracker(tmp_path, fake_runner("0,1,10,20\n1,0,0,0\n")).track(video)
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


def test_uppercase_extension_is_converted_to_mp4(tmp_path):
    # predict.py asserts a lowercase '.mp4' suffix; phones write .MP4 / .MOV.
    seen = []
    video = make_video(tmp_path / "phone clip.MOV", n_frames=6)
    df = make_tracker(tmp_path, fake_runner("0,1,10,20\n", seen)).track(video)
    prepared, meta = seen[0]
    assert prepared.suffix == ".mp4" and meta.n_frames == 6 and df["visible"].tolist() == [True]


def test_large_video_is_downscaled_and_coords_scaled_back(tmp_path):
    # predict.py holds every full-size frame in RAM, so 1080p+ clips are shrunk to 720p first.
    seen = []
    video = make_video(tmp_path / "big.mp4", n_frames=5, size=(1920, 1080))
    df = make_tracker(tmp_path, fake_runner("0,1,100,50\n", seen)).track(video)
    _, meta = seen[0]
    assert (meta.width, meta.height, meta.n_frames) == (1280, 720, 5)
    assert (df.loc[0, "x"], df.loc[0, "y"]) == pytest.approx((150.0, 75.0))


def test_small_mp4_is_passed_through(tmp_path):
    seen = []
    video = make_video(tmp_path / "small.mp4", n_frames=5)
    make_tracker(tmp_path, fake_runner("0,1,10,20\n", seen)).track(video)
    assert seen[0][0] == video.resolve()


def test_missing_output_raises(tmp_path):
    with pytest.raises(RuntimeError, match="_ball.csv"):
        make_tracker(tmp_path, lambda cmd, cwd: None).track(make_video(tmp_path / "x.mp4", n_frames=5))
