import json

import numpy as np
import pandas as pd
import pytest

from badminton_analyzer.schema import empty_shots_table
from badminton_analyzer.shots import build_shots_table
from badminton_analyzer.stats import rally_stats
from tests.helpers import build_frames


def walking_frames(n=91):
    f = build_frames(n)
    f["near_detected"] = True
    f["near_court_x"] = 3.0
    f["near_court_y"] = np.linspace(8.0, 11.0, n)
    return f


def test_distance_covered_straight_walk():
    s = rally_stats(walking_frames(), empty_shots_table(), fps=30.0)
    assert s["players"]["near"]["distance_m"] == pytest.approx(3.0, abs=0.05)


def test_detection_spike_is_ignored():
    f = walking_frames()
    f.loc[45, "near_court_y"] += 5.0
    assert rally_stats(f, empty_shots_table(), 30.0)["players"]["near"]["distance_m"] == pytest.approx(3.0, abs=0.05)


def test_no_shots_is_json_safe():
    s = rally_stats(build_frames(30), empty_shots_table(), 30.0)
    assert s["n_shots"] == 0 and s["shot_counts"] == {} and s["players"]["far"]["distance_m"] == 0.0
    json.dumps(s, allow_nan=False)


def test_stats_shot_times():
    shots = build_shots_table(
        build_frames(60), pd.DataFrame({"hit_frame": [0, 30], "hitter": ["far", "near"]}), ["unknown"] * 2
    )
    s = rally_stats(build_frames(60), shots, 30.0)
    assert s["shot_counts"] == {"unknown": 2} and [x["time_s"] for x in s["shots"]] == [0.0, 1.0]
    json.dumps(s, allow_nan=False)
