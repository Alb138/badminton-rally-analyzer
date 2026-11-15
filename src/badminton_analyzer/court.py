"""Court geometry and the image-to-court homography from four clicked corners.

Court coordinates are metres: origin at the far-left outer (doubles) corner as
seen in the video, x to the right (0-6.10), y toward the camera (0-13.40).
"""
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COURT_LENGTH_M = 13.40
COURT_WIDTH_M = 6.10
SINGLES_WIDTH_M = 5.18
NET_Y_M = 6.70
SHORT_SERVICE_FROM_NET_M = 1.98

COURT_CORNERS_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)
MIN_CORNER_AREA_PX = 100.0


@dataclass(frozen=True)
class CourtCalibration:
    image_points: np.ndarray  # (4, 2): far_left, far_right, near_right, near_left
    homography: np.ndarray  # (3, 3): image -> court


def order_corners(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError(f"need exactly 4 court corners, got {len(pts)}")
    by_y = pts[np.argsort(pts[:, 1], kind="stable")]
    far = by_y[:2][np.argsort(by_y[:2, 0], kind="stable")]
    near = by_y[2:][np.argsort(-by_y[2:, 0], kind="stable")]
    ordered = np.vstack([far, near])
    x, y = ordered[:, 0], ordered[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < MIN_CORNER_AREA_PX:
        raise ValueError("court corners are duplicated or collinear; click the 4 outer corners")
    return ordered


def calibrate(points: Sequence[tuple[float, float]]) -> CourtCalibration:
    image_points = order_corners(points)
    homography = cv2.getPerspectiveTransform(image_points.astype(np.float32), COURT_CORNERS_M)
    return CourtCalibration(image_points=image_points, homography=homography)


def image_to_court(cal: CourtCalibration, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = np.isfinite(pts).all(axis=1)
    if ok.any():
        mapped = cv2.perspectiveTransform(pts[ok].reshape(-1, 1, 2), cal.homography)
        out[ok] = mapped.reshape(-1, 2)
    return out


def save_calibration(cal: CourtCalibration, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"image_points": cal.image_points.tolist()}, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> CourtCalibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return calibrate([tuple(p) for p in data["image_points"]])
