from badminton_analyzer.hits import HitParams, detect_hits
from tests.helpers import build_frames, polyline

NEAR, FAR = (600, 450, 680, 610), (610, 150, 670, 270)


def hits_of(shuttle, n, fps=30.0, near=NEAR, far=FAR):
    h = detect_hits(build_frames(n, shuttle, near, far), fps, 720, HitParams())
    return list(zip(h["hit_frame"], h["hitter"]))


def test_serve_and_return_detected():
    got = hits_of(polyline([(0, (640, 200)), (30, (640, 470)), (60, (640, 200))]), 61)
    assert got[0] == (0, "far") and got[1][1] == "near" and abs(got[1][0] - 30) <= 1 and len(got) == 2


def test_turn_far_from_players_is_ignored():
    shuttle = polyline([(0, (640, 200)), (30, (640, 470)), (60, (640, 200))])
    assert hits_of(shuttle, 61, near=(100, 450, 180, 610), far=(100, 150, 180, 270)) == []


def test_all_invisible_returns_empty_table():
    h = detect_hits(build_frames(30, None, NEAR, FAR), 30.0, 720, HitParams())
    assert h.empty and list(h.columns) == ["hit_frame", "hitter"]


def test_no_hit_inferred_across_tracking_gap():
    down = polyline([(0, (640, 200)), (30, (640, 470))])
    up = polyline([(51, (640, 470)), (80, (640, 200))])
    assert hits_of({**down, **up}, 81) == [(0, "far")]


def test_same_hitter_twice_keeps_sharper_turn():
    shuttle = polyline([(0, (640, 200)), (30, (640, 470)), (45, (660, 430)), (60, (640, 470)), (90, (640, 200))])
    got = hits_of(shuttle, 91)
    assert got[0] == (0, "far") and len(got) == 2 and got[1][1] == "near" and abs(got[1][0] - 45) <= 1


def test_thresholds_scale_with_fps():
    got = hits_of(polyline([(0, (640, 200)), (60, (640, 470)), (120, (640, 200))]), 121, fps=60.0)
    assert len(got) == 2 and abs(got[1][0] - 60) <= 2
