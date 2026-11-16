import json

import pandas as pd
import pytest

from badminton_analyzer import pipeline
from badminton_analyzer.court import calibrate, save_calibration
from badminton_analyzer.pipeline import load_pipeline_config, run_pipeline
from badminton_analyzer.schema import validate_frames, validate_shots
from badminton_analyzer.shots import UnknownShotClassifier
from badminton_analyzer.video_io import read_video_meta
from tests.helpers import CLICKS_160, CONFIG, FakePose, FakeTracker, make_video


def run(tmp_path, tracker):
    video = make_video(tmp_path / "my clips" / "rally 1.mp4", n_frames=60)
    return run_pipeline(video, calibrate(CLICKS_160), tmp_path / "out dir", tracker, FakePose(),
                        UnknownShotClassifier(), load_pipeline_config(CONFIG))


def test_pipeline_writes_all_outputs(tmp_path):
    out = run(tmp_path, FakeTracker())
    validate_frames(pd.read_parquet(out.frames_path))
    validate_shots(pd.read_parquet(out.shots_path))
    stats = json.loads(out.stats_path.read_text())
    assert stats["n_frames"] == 60 and stats["n_shots"] >= 1
    assert read_video_meta(out.video_path).n_frames == 60


def test_tracker_output_longer_than_video_is_clipped(tmp_path):
    out = run(tmp_path, FakeTracker(n=70))
    assert len(pd.read_parquet(out.frames_path)) == 60


def test_tracker_failure_stops_before_pose(tmp_path):
    # The tracker is the stage most likely to reject an input; fail before the long pose pass.
    class FailingTracker:
        def track(self, video_path):
            raise RuntimeError("tracker rejected the clip")

    class CountingPose(FakePose):
        calls = 0

        def detect(self, frame):
            CountingPose.calls += 1
            return super().detect(frame)

    video = make_video(tmp_path / "v.mp4", n_frames=10)
    with pytest.raises(RuntimeError, match="rejected"):
        run_pipeline(video, calibrate(CLICKS_160), tmp_path / "o", FailingTracker(), CountingPose(),
                     UnknownShotClassifier(), load_pipeline_config(CONFIG))
    assert CountingPose.calls == 0


def test_cli_main(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "build_default_components",
                        lambda cfg: (FakeTracker(), FakePose(), UnknownShotClassifier()))
    video = make_video(tmp_path / "v.mp4", n_frames=60)
    save_calibration(calibrate(CLICKS_160), tmp_path / "cal.json")
    rc = pipeline.main([str(video), "--calib", str(tmp_path / "cal.json"), "--out", str(tmp_path / "o"),
                        "--config", str(CONFIG)])
    assert rc == 0 and (tmp_path / "o" / "annotated.mp4").exists()
