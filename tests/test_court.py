import numpy as np
import pytest

from badminton_analyzer.court import calibrate, image_to_court, load_calibration, save_calibration
from tests.helpers import CLICKS_720

COURT = [[0, 0], [6.10, 0], [6.10, 13.40], [0, 13.40]]


def test_corners_map_to_court_corners():
    out = image_to_court(calibrate(CLICKS_720), np.array(CLICKS_720, dtype=float))
    np.testing.assert_allclose(out, COURT, atol=1e-6)


def test_click_order_does_not_matter():
    shuffled = [CLICKS_720[i] for i in (2, 0, 3, 1)]
    np.testing.assert_allclose(calibrate(shuffled).homography, calibrate(CLICKS_720).homography)


def test_collinear_corners_rejected():
    with pytest.raises(ValueError, match="corners"):
        calibrate([(0, 0), (50, 0), (100, 0), (150, 0)])


def test_duplicated_corner_rejected():
    # A double-click on one corner must not yield a silently useless homography.
    with pytest.raises(ValueError, match="corners"):
        calibrate([CLICKS_720[0], CLICKS_720[0], CLICKS_720[2], CLICKS_720[3]])


def test_three_collinear_corners_rejected():
    with pytest.raises(ValueError, match="corners"):
        calibrate([(400, 200), (640, 200), (880, 200), (640, 650)])


def test_wrong_number_of_points_rejected():
    with pytest.raises(ValueError, match="4"):
        calibrate(CLICKS_720[:3])


def test_nan_points_pass_through():
    out = image_to_court(calibrate(CLICKS_720), np.array([[np.nan, np.nan], [640.0, 420.0]]))
    assert np.isnan(out[0]).all() and np.isfinite(out[1]).all()


def test_calibration_roundtrip(tmp_path):
    cal = calibrate(CLICKS_720)
    save_calibration(cal, tmp_path / "c d" / "cal.json")
    np.testing.assert_allclose(load_calibration(tmp_path / "c d" / "cal.json").homography, cal.homography)
