import numpy as np
import pandas as pd
import pytest

from badminton_analyzer.data.nycu import NycuRally, read_ball_csv
from badminton_analyzer.tracking.evaluate import Counts, count_outcomes, evaluate_tracker, scores, tolerance_px
from tests.helpers import make_video


def table(rows):  # rows: (frame, visible, x, y)
    return pd.DataFrame(rows, columns=["frame", "visible", "x", "y"])


def test_count_outcomes_follows_tracknet_rules():
    gt = table([(0, True, 100, 100), (1, True, 100, 100), (2, False, np.nan, np.nan), (3, True, 50, 50),
                (4, False, np.nan, np.nan)])
    pred = table([(0, True, 105, 105), (1, True, 120, 100), (2, True, 10, 10), (3, False, np.nan, np.nan),
                  (4, False, np.nan, np.nan)])
    assert count_outcomes(gt, pred, tolerance=10.0) == Counts(tp=1, fp1=1, fp2=1, fn=1, tn=1)


def test_missing_predictions_count_as_invisible_and_extra_rows_are_ignored():
    gt = table([(0, True, 10, 10), (1, True, 10, 10)])
    pred = table([(0, True, 10, 10), (5, True, 1, 1)])
    assert count_outcomes(gt, pred, 10.0) == Counts(tp=1, fn=1)


def test_scores_use_tracknet_formulas():
    s = scores(Counts(tp=8, tn=5, fp1=1, fp2=1, fn=2))
    assert s["precision"] == pytest.approx(0.8) and s["recall"] == pytest.approx(0.8)
    assert s["f1"] == pytest.approx(0.8) and s["accuracy"] == pytest.approx(13 / 17)
    assert scores(Counts())["f1"] == 0.0


def test_tolerance_scales_with_width():
    assert tolerance_px(512) == 4.0 and tolerance_px(1280) == 10.0 and tolerance_px(1920) == 15.0


def test_evaluate_tracker_scores_a_perfect_tracker(tmp_path):
    video = make_video(tmp_path / "v.mp4", n_frames=6)
    labels = tmp_path / "1_01_00_ball.csv"
    labels.write_text("Frame,Visibility,X,Y\n" + "".join(
        f"{i},{0 if i == 2 else 1},{0 if i == 2 else 40},{0 if i == 2 else 20}\n" for i in range(6)))
    rally = NycuRally("Test/match1", "1_01_00", video, labels)

    class Perfect:
        def track(self, video_path):
            return read_ball_csv(labels)

    [result] = evaluate_tracker(Perfect(), [rally])
    assert result.counts == Counts(tp=5, tn=1) and result.n_frames == 6 and result.seconds >= 0
