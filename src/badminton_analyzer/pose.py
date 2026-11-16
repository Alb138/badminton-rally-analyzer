"""Player detection with YOLOv8-pose and assignment to the near/far court halves."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from badminton_analyzer.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, CourtCalibration, image_to_court
from badminton_analyzer.schema import KPT_LEN, PLAYER_IDS, empty_frames_table
from badminton_analyzer.video_io import iter_frames

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
ANKLE_MIN_CONF = 0.3


@dataclass(frozen=True)
class PersonDetection:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    kpts: np.ndarray  # (17, 3) x, y, conf (COCO order)
    conf: float


class PoseEstimator(Protocol):
    def detect(self, frame: np.ndarray) -> list[PersonDetection]: ...


class YoloPoseEstimator:
    def __init__(self, weights: str = "yolov8m-pose.pt", conf: float = 0.3, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        result = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()
        return [PersonDetection(boxes[i], kpts[i], float(confs[i])) for i in range(len(boxes))]


def foot_point(det: PersonDetection) -> np.ndarray:
    ankles = det.kpts[[LEFT_ANKLE, RIGHT_ANKLE]]
    if (ankles[:, 2] >= ANKLE_MIN_CONF).all():
        return ankles[:, :2].mean(axis=0)
    x1, _, x2, y2 = det.bbox
    return np.array([(x1 + x2) / 2, y2], dtype=float)


def assign_players(
    dets: list[PersonDetection], cal: CourtCalibration, side_margin_m: float, end_margin_m: float
) -> dict[str, PersonDetection | None]:
    best: dict[str, PersonDetection | None] = {pid: None for pid in PLAYER_IDS}
    if not dets:
        return best
    court = image_to_court(cal, np.array([foot_point(d) for d in dets]))
    for det, (cx, cy) in zip(dets, court):
        on_court = (-side_margin_m <= cx <= COURT_WIDTH_M + side_margin_m) and (
            -end_margin_m <= cy <= COURT_LENGTH_M + end_margin_m
        )
        if not on_court:
            continue
        half = "far" if cy < NET_Y_M else "near"
        if best[half] is None or det.conf > best[half].conf:
            best[half] = det
    return best


def extract_players(
    video_path: Path,
    estimator: PoseEstimator,
    cal: CourtCalibration,
    side_margin_m: float,
    end_margin_m: float,
) -> pd.DataFrame:
    per_frame = [
        assign_players(estimator.detect(frame), cal, side_margin_m, end_margin_m) for frame in iter_frames(video_path)
    ]
    df = empty_frames_table(len(per_frame))
    for pid in PLAYER_IDS:
        rows = [(i, a[pid]) for i, a in enumerate(per_frame) if a[pid] is not None]
        if not rows:
            continue
        idx = [i for i, _ in rows]
        dets = [d for _, d in rows]
        feet = np.array([foot_point(d) for d in dets])
        court = image_to_court(cal, feet)
        boxes = np.array([d.bbox for d in dets], dtype=float)
        df.loc[idx, f"{pid}_detected"] = True
        for j, name in enumerate(("x1", "y1", "x2", "y2")):
            df.loc[idx, f"{pid}_{name}"] = boxes[:, j]
        df.loc[idx, f"{pid}_foot_x"] = feet[:, 0]
        df.loc[idx, f"{pid}_foot_y"] = feet[:, 1]
        df.loc[idx, f"{pid}_court_x"] = court[:, 0]
        df.loc[idx, f"{pid}_court_y"] = court[:, 1]
        kpts_col = df[f"{pid}_kpts"].tolist()
        for i, d in zip(idx, dets):
            kpts_col[i] = np.asarray(d.kpts, dtype=float).reshape(-1)[:KPT_LEN].tolist()
        df[f"{pid}_kpts"] = kpts_col
    return df
