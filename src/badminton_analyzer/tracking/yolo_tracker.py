"""Single-frame shuttle tracker: a YOLOv8 detector run on every frame, best box wins."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from badminton_analyzer.video_io import iter_frames


class YoloShuttleTracker:
    """``predict_fn(frame) -> (N, 3)`` array of [cx, cy, conf] in source pixels; the default wraps Ultralytics."""

    def __init__(self, weights: Path | None, conf: float = 0.25, imgsz: int = 1280, device: str | None = None,
                 predict_fn: Callable[[np.ndarray], np.ndarray] | None = None):
        if predict_fn is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))

            def predict_fn(frame: np.ndarray) -> np.ndarray:
                boxes = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0].boxes
                if boxes is None or len(boxes) == 0:
                    return np.zeros((0, 3))
                xywh = boxes.xywh.cpu().numpy()
                return np.column_stack([xywh[:, 0], xywh[:, 1], boxes.conf.cpu().numpy()])

        self.predict_fn = predict_fn

    def track(self, video_path: Path) -> pd.DataFrame:
        rows = []
        for i, frame in enumerate(iter_frames(Path(video_path))):
            boxes = np.asarray(self.predict_fn(frame), dtype=float).reshape(-1, 3)
            if len(boxes):
                cx, cy, _ = boxes[int(np.argmax(boxes[:, 2]))]
                rows.append((i, True, float(cx), float(cy)))
            else:
                rows.append((i, False, np.nan, np.nan))
        return pd.DataFrame(rows, columns=["frame", "visible", "x", "y"])
