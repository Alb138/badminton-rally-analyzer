from pathlib import Path

import pandas as pd

from badminton_analyzer.schema import validate_shots
from badminton_analyzer.shots import UnknownShotClassifier, build_shots_table
from tests.helpers import build_frames


def test_build_shots_table_reads_hitter_position():
    f = build_frames(40)
    f.loc[30, ["near_detected", "near_court_x", "near_court_y"]] = [True, 3.0, 11.0]
    shots = build_shots_table(f, pd.DataFrame({"hit_frame": [30], "hitter": ["near"]}), ["unknown"])
    validate_shots(shots)
    assert shots.iloc[0][["hitter_court_x", "hitter_court_y"]].tolist() == [3.0, 11.0]


def test_build_shots_table_with_no_hits_is_valid():
    shots = build_shots_table(build_frames(10), pd.DataFrame({"hit_frame": [], "hitter": []}), [])
    validate_shots(shots)
    assert shots.empty


def test_unknown_classifier_labels_every_hit():
    hits = pd.DataFrame({"hit_frame": [0, 30], "hitter": ["far", "near"]})
    assert UnknownShotClassifier().classify(Path("v.mp4"), build_frames(40), hits) == ["unknown", "unknown"]
