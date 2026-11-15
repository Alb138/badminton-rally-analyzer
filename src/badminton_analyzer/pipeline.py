"""End-to-end pipeline: rally clip -> frames/shots tables, stats JSON and annotated video.

CLI: badminton-analyze VIDEO --calib CAL.json [--out DIR] [--config configs/pipeline.yaml]
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from badminton_analyzer.court import CourtCalibration, load_calibration
from badminton_analyzer.hits import HitParams, detect_hits
from badminton_analyzer.pose import PoseEstimator, YoloPoseEstimator, extract_players
from badminton_analyzer.render import RenderParams, render_video
from badminton_analyzer.schema import validate_frames, validate_shots
from badminton_analyzer.shots import ShotClassifier, UnknownShotClassifier, build_shots_table
from badminton_analyzer.stats import rally_stats
from badminton_analyzer.tracking.base import ShuttleTracker
from badminton_analyzer.tracking.tracknetv3 import TrackNetV3Tracker
from badminton_analyzer.video_io import read_video_meta


@dataclass(frozen=True)
class PipelineConfig:
    tracker: dict
    pose: dict
    hits: HitParams
    render: RenderParams


@dataclass(frozen=True)
class PipelineOutputs:
    frames_path: Path
    shots_path: Path
    stats_path: Path
    video_path: Path


def load_pipeline_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PipelineConfig(
        tracker=raw["tracker"],
        pose=raw["pose"],
        hits=HitParams(**raw["hits"]),
        render=RenderParams(**raw["render"]),
    )


def build_default_components(cfg: PipelineConfig) -> tuple[ShuttleTracker, PoseEstimator, ShotClassifier]:
    t = cfg.tracker
    if t.get("kind") != "tracknetv3":
        raise ValueError(f"unknown tracker kind: {t.get('kind')}")
    tracker = TrackNetV3Tracker(
        repo_dir=Path(t["repo_dir"]).resolve(),
        tracknet_ckpt=Path(t["tracknet_ckpt"]).resolve(),
        inpaintnet_ckpt=Path(t["inpaintnet_ckpt"]).resolve(),
        work_dir=Path(t["work_dir"]).resolve(),
        batch_size=int(t["batch_size"]),
    )
    pose = YoloPoseEstimator(cfg.pose["weights"], conf=float(cfg.pose["conf"]))
    return tracker, pose, UnknownShotClassifier()


def _merge_shuttle(frames: pd.DataFrame, shuttle: pd.DataFrame) -> pd.DataFrame:
    """Copy tracker output into the shuttle_* columns; rows past the decoded frames are dropped."""
    s = shuttle[(shuttle["frame"] >= 0) & (shuttle["frame"] < len(frames))]
    idx = s["frame"].to_numpy()
    frames.loc[idx, "shuttle_visible"] = s["visible"].astype(bool).to_numpy()
    frames.loc[idx, "shuttle_x"] = s["x"].to_numpy(dtype=float)
    frames.loc[idx, "shuttle_y"] = s["y"].to_numpy(dtype=float)
    return frames


def run_pipeline(
    video_path: Path,
    calibration: CourtCalibration,
    out_dir: Path,
    tracker: ShuttleTracker,
    pose: PoseEstimator,
    shot_classifier: ShotClassifier,
    cfg: PipelineConfig,
) -> PipelineOutputs:
    video_path, out_dir = Path(video_path), Path(out_dir)
    meta = read_video_meta(video_path)
    fps = meta.fps or 30.0

    frames = extract_players(
        video_path, pose, calibration, float(cfg.pose["side_margin_m"]), float(cfg.pose["end_margin_m"])
    )
    frames = _merge_shuttle(frames, tracker.track(video_path))
    validate_frames(frames)
    hits = detect_hits(frames, fps, meta.height, cfg.hits)
    shots = build_shots_table(frames, hits, shot_classifier.classify(video_path, frames, hits))
    validate_shots(shots)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = PipelineOutputs(
        frames_path=out_dir / "frames.parquet",
        shots_path=out_dir / "shots.parquet",
        stats_path=out_dir / "stats.json",
        video_path=out_dir / "annotated.mp4",
    )
    frames.to_parquet(outputs.frames_path, index=False)
    shots.to_parquet(outputs.shots_path, index=False)
    stats = rally_stats(frames, shots, fps)
    outputs.stats_path.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_video(video_path, frames, shots, outputs.video_path, cfg.render)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="badminton-analyze", description="Analyze one singles rally clip.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--calib", type=Path, required=True, help="court calibration JSON (scripts/calibrate_court.py)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: outputs/<video stem>)")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    args = parser.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    tracker, pose, classifier = build_default_components(cfg)
    out_dir = args.out or Path("outputs") / args.video.stem
    outputs = run_pipeline(args.video, load_calibration(args.calib), out_dir, tracker, pose, classifier, cfg)
    for path in (outputs.video_path, outputs.stats_path, outputs.frames_path, outputs.shots_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
