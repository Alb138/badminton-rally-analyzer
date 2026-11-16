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


def noisy_walk(fps, seconds=3.0, noise_m=0.08, seed=0):
    n = int(fps * seconds) + 1
    rng = np.random.default_rng(seed)
    f = build_frames(n)
    f["near_detected"] = True
    f["near_court_x"] = 3.0 + rng.normal(0, noise_m, n)
    f["near_court_y"] = np.linspace(8.0, 11.0, n) + rng.normal(0, noise_m, n)
    return f


def test_distance_does_not_depend_on_fps():
    # Foot-position noise must not add up faster at 60 fps than at 30 fps (phone footage is often 60 fps).
    # Averaged over fixed seeds: one noisy 3 m walk varies by about +-0.12 m on its own.
    def mean_distance(fps):
        runs = [rally_stats(noisy_walk(fps, seed=s), empty_shots_table(), float(fps)) for s in range(10)]
        return float(np.mean([r["players"]["near"]["distance_m"] for r in runs]))

    d30, d60 = mean_distance(30), mean_distance(60)
    assert abs(d30 - 3.0) < 0.25 and abs(d60 - 3.0) < 0.25 and abs(d30 - d60) < 0.15


def test_sprint_at_low_fps_is_counted():
    # 7 m/s for 1 s filmed at 12 fps: per-frame steps of 0.58 m are real movement, not detection jumps.
    f = build_frames(13)
    f["near_detected"] = True
    f["near_court_x"] = 3.0
    f["near_court_y"] = np.linspace(4.0, 11.0, 13)
    assert rally_stats(f, empty_shots_table(), 12.0)["players"]["near"]["distance_m"] == pytest.approx(7.0, abs=0.3)
