"""YOLO detection dataset from NYCU's per-frame shuttle points: one 16x16 px box per visible shuttle."""
import math
from pathlib import Path

import cv2
import yaml

from badminton_analyzer.data.nycu import NycuRally, read_ball_csv
from badminton_analyzer.video_io import iter_frames

BOX_PX = 16  # at the source resolution; a broadcast shuttle is roughly 8-12 px


def yolo_label_line(x: float, y: float, width: int, height: int, box_px: int = BOX_PX) -> str:
    """'0 cx cy w h' normalized to the image, with the box clipped to the image edges."""
    half = box_px / 2
    x1, x2 = max(0.0, x - half), min(float(width), x + half)
    y1, y2 = max(0.0, y - half), min(float(height), y + half)
    cx, w = _fit((x1 + x2) / 2 / width, (x2 - x1) / width)
    cy, h = _fit((y1 + y2) / 2 / height, (y2 - y1) / height)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _fit(centre: float, size: float) -> tuple[float, float]:
    """Round to 6 decimals without the box edge crossing 0 or 1 (size is rounded down)."""
    c = round(centre, 6)
    s = math.floor(min(size, 2 * c, 2 * (1 - c)) * 1e6) / 1e6
    return c, s


def build_yolo_dataset(rallies: dict[str, list[NycuRally]], out_root: Path, stride: int = 3, quality: int = 90) -> Path:
    """Write images/<split>/*.jpg + labels/<split>/*.txt (empty for no shuttle) and data.yaml.

    data.yaml has no 'path' key, so Ultralytics resolves the folders next to it wherever the
    dataset is copied (locally or a Kaggle input folder).
    """
    out_root = Path(out_root)
    for split, split_rallies in rallies.items():
        image_dir, label_dir = out_root / "images" / split, out_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for rally in split_rallies:
            labels = read_ball_csv(rally.label_path).set_index("frame")
            prefix = f"{rally.match_id.replace('/', '-')}__{rally.rally_id}"
            for i, frame in enumerate(iter_frames(Path(rally.video_path))):
                if i % stride:
                    continue
                name = f"{prefix}__{i:05d}"
                cv2.imwrite(str(image_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                visible = i in labels.index and bool(labels.at[i, "visible"])
                line = yolo_label_line(labels.at[i, "x"], labels.at[i, "y"], frame.shape[1], frame.shape[0]) if visible else ""
                (label_dir / f"{name}.txt").write_text(line + ("\n" if line else ""), encoding="utf-8")
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": {0: "shuttle"}}),
                         encoding="utf-8")
    return data_yaml
