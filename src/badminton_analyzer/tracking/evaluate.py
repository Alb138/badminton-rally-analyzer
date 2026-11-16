"""Score shuttle trackers against labels with the TrackNetV2/V3 protocol.

Each frame is one of TP (both visible, within tolerance), FP1 (both visible, too far),
FP2 (predicted visible, label invisible), FN (predicted invisible, label visible) or
TN (both invisible). Precision = TP/(TP+FP1+FP2), recall = TP/(TP+FN). The tolerance is
4 px in a 512-px-wide frame (TrackNet's input), scaled to the video's width.
"""
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from badminton_analyzer.data.nycu import NycuRally, read_ball_csv
from badminton_analyzer.video_io import read_video_meta

TOLERANCE_AT_512 = 4.0


def tolerance_px(frame_width: int) -> float:
    return TOLERANCE_AT_512 * frame_width / 512


@dataclass
class Counts:
    tp: int = 0
    tn: int = 0
    fp1: int = 0
    fp2: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})


def count_outcomes(gt: pd.DataFrame, pred: pd.DataFrame, tolerance: float) -> Counts:
    """Classify every labelled frame; frames missing from ``pred`` count as invisible."""
    p = pred.drop_duplicates("frame").set_index("frame")[["visible", "x", "y"]]
    m = gt[["frame", "visible", "x", "y"]].merge(p, left_on="frame", right_index=True, how="left", suffixes=("", "_p"))
    gv = m["visible"].astype(bool).to_numpy()
    pv = m["visible_p"].eq(True).to_numpy()
    dist = np.hypot(m["x"] - m["x_p"], m["y"] - m["y_p"]).to_numpy()
    close = np.nan_to_num(dist, nan=np.inf) <= tolerance
    return Counts(
        tp=int((gv & pv & close).sum()),
        tn=int((~gv & ~pv).sum()),
        fp1=int((gv & pv & ~close).sum()),
        fp2=int((~gv & pv).sum()),
        fn=int((gv & ~pv).sum()),
    )


def scores(c: Counts) -> dict[str, float]:
    total = c.tp + c.tn + c.fp1 + c.fp2 + c.fn
    precision = c.tp / (c.tp + c.fp1 + c.fp2) if c.tp + c.fp1 + c.fp2 else 0.0
    recall = c.tp / (c.tp + c.fn) if c.tp + c.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (c.tp + c.tn) / total if total else 0.0}


@dataclass
class RallyResult:
    match_id: str
    rally_id: str
    counts: Counts
    n_frames: int
    seconds: float


def evaluate_tracker(tracker, rallies: list[NycuRally], progress=None) -> list[RallyResult]:
    """Run ``tracker`` on each rally and score it. ``progress(i, n, rally)`` is called before each rally."""
    results = []
    for i, rally in enumerate(rallies):
        if progress:
            progress(i, len(rallies), rally)
        width = read_video_meta(Path(rally.video_path)).width
        gt = read_ball_csv(rally.label_path)
        start = time.perf_counter()
        pred = tracker.track(Path(rally.video_path))
        seconds = time.perf_counter() - start
        results.append(RallyResult(rally.match_id, rally.rally_id, count_outcomes(gt, pred, tolerance_px(width)),
                                   len(gt), seconds))
    return results


def summarize(results: list[RallyResult]) -> dict:
    total = sum((r.counts for r in results), Counts())
    frames = sum(r.n_frames for r in results)
    seconds = sum(r.seconds for r in results)
    return {
        "counts": total.__dict__.copy(),
        **scores(total),
        "rallies": len(results),
        "frames": frames,
        "seconds": seconds,
        "fps": frames / seconds if seconds else 0.0,
    }
