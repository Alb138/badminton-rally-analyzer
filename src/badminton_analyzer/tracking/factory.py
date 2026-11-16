"""Build a shuttle tracker from a config section (configs/pipeline.yaml or configs/trackers.yaml)."""
from collections.abc import Callable
from pathlib import Path

from badminton_analyzer.tracking.base import ShuttleTracker
from badminton_analyzer.tracking.tracknetv3 import TrackNetV3Tracker


def _tracknetv3(cfg: dict) -> ShuttleTracker:
    return TrackNetV3Tracker(
        repo_dir=Path(cfg["repo_dir"]).resolve(),
        tracknet_ckpt=Path(cfg["tracknet_ckpt"]).resolve(),
        inpaintnet_ckpt=Path(cfg["inpaintnet_ckpt"]).resolve(),
        work_dir=Path(cfg["work_dir"]).resolve(),
        batch_size=int(cfg.get("batch_size", 2)),
    )


def _existing(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tracker weights not found: {p.as_posix()}")
    return p


def _tracknet(cfg: dict) -> ShuttleTracker:
    from badminton_analyzer.tracking.tracknet_tracker import TrackNetTracker

    return TrackNetTracker(_existing(cfg["weights"]), device=cfg.get("device"),
                           batch_size=int(cfg.get("batch_size", 4)), threshold=float(cfg.get("threshold", 0.5)))


def _yolo(cfg: dict) -> ShuttleTracker:
    from badminton_analyzer.tracking.yolo_tracker import YoloShuttleTracker

    return YoloShuttleTracker(_existing(cfg["weights"]), conf=float(cfg.get("conf", 0.25)),
                              imgsz=int(cfg.get("imgsz", 1280)), device=cfg.get("device"))


BUILDERS: dict[str, Callable[[dict], ShuttleTracker]] = {
    "tracknetv3": _tracknetv3,
    "tracknet": _tracknet,
    "yolo": _yolo,
}


def make_tracker(cfg: dict) -> ShuttleTracker:
    kind = cfg.get("kind")
    if kind not in BUILDERS:
        raise ValueError(f"unknown tracker kind: {kind!r} (known: {', '.join(sorted(BUILDERS))})")
    return BUILDERS[kind](cfg)
