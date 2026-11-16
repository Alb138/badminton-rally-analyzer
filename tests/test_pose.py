import numpy as np

from badminton_analyzer.court import calibrate
from badminton_analyzer.pose import PersonDetection, assign_players, extract_players, foot_point
from tests.helpers import CLICKS_160, CLICKS_720, make_video

CAL = calibrate(CLICKS_720)


def det(foot, conf, ankle_conf=0.9):
    fx, fy = foot
    kpts = np.zeros((17, 3))
    kpts[15] = [fx - 10, fy, ankle_conf]
    kpts[16] = [fx + 10, fy, ankle_conf]
    return PersonDetection(np.array([fx - 30, fy - 160, fx + 30, fy], dtype=float), kpts, conf)


def foot(d):
    return tuple(int(v) for v in np.round(foot_point(d)))


def test_assigns_near_far_and_drops_umpire():
    out = assign_players([det((640, 600), .9), det((640, 260), .8), det((60, 420), .95)], CAL, 0.75, 2.0)
    assert foot(out["near"]) == (640, 600) and foot(out["far"]) == (640, 260)


def test_keeps_most_confident_per_half():
    out = assign_players([det((600, 600), .5), det((700, 600), .9)], CAL, 0.75, 2.0)
    assert foot(out["near"]) == (700, 600) and out["far"] is None


def test_foot_point_falls_back_to_bbox_bottom():
    d = det((640, 600), .9, ankle_conf=0.1)
    np.testing.assert_allclose(foot_point(d), [(d.bbox[0] + d.bbox[2]) / 2, d.bbox[3]])


def test_extract_players_fills_every_frame(tmp_path):
    class NearOnly:
        def detect(self, frame):
            return [det((80, 86), .9)]

    df = extract_players(make_video(tmp_path / "v.mp4", n_frames=10), NearOnly(), calibrate(CLICKS_160), 0.75, 2.0)
    assert len(df) == 10 and df["near_detected"].all() and not df["far_detected"].any()
    assert df["far_court_x"].isna().all() and (df["near_court_y"] > 6.70).all()
