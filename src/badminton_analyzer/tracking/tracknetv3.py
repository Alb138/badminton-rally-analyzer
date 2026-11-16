"""Pretrained TrackNetV3 (third_party/TrackNetV3) run as a subprocess.

This is the Phase 1 stand-in tracker; the project's own trackers replace it in Phase 2.
"""
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pandas as pd

from badminton_analyzer.data.nycu import read_ball_csv
from badminton_analyzer.tracking import tracknetv3_launcher
from badminton_analyzer.video_io import read_video_meta

LAUNCHER = Path(tracknetv3_launcher.__file__)
# predict.py keeps every full-size frame in RAM and works at 512x288 internally,
# so anything taller than 720p is shrunk first.
MAX_TRACK_HEIGHT = 720


def prepare_video(video_path: Path, out_dir: Path, max_height: int = MAX_TRACK_HEIGHT) -> tuple[Path, float, float]:
    """Return a clip predict.py accepts plus the (x, y) factors that map its coordinates back.

    predict.py asserts a lowercase '.mp4' suffix, so other containers (.MP4, .MOV, .mkv)
    and tall clips are re-encoded to ``out_dir/<stem>.mp4``, keeping every frame.
    """
    video_path = Path(video_path)
    meta = read_video_meta(video_path)
    if video_path.suffix == ".mp4" and meta.height <= max_height:
        return video_path, 1.0, 1.0
    out = Path(out_dir) / f"{video_path.stem}.mp4"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_path),
        "-map", "0:v:0", "-vf", f"scale=-2:{min(meta.height, max_height)}", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not convert {video_path.name} for TrackNetV3:\n{exc.stderr.strip()}") from exc
    new = read_video_meta(out)
    return out, meta.width / new.width, meta.height / new.height


def run_subprocess(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join((exc.stderr or "").strip().splitlines()[-20:])
        raise RuntimeError(f"TrackNetV3 failed (exit {exc.returncode}):\n{tail}") from exc


@dataclass
class TrackNetV3Tracker:
    repo_dir: Path
    tracknet_ckpt: Path
    inpaintnet_ckpt: Path
    work_dir: Path
    batch_size: int = 16
    runner: Callable[[list[str], Path], None] = run_subprocess

    def build_command(self, video_path: Path, save_dir: Path) -> list[str]:
        # predict.py names its output with video_file.split('/'), so pass forward slashes.
        return [
            sys.executable, str(LAUNCHER), "predict.py",
            "--video_file", Path(video_path).as_posix(),
            "--tracknet_file", str(self.tracknet_ckpt),
            "--inpaintnet_file", str(self.inpaintnet_ckpt),
            "--save_dir", str(save_dir),
            "--batch_size", str(self.batch_size),
        ]

    def track(self, video_path: Path) -> pd.DataFrame:
        video_path = Path(video_path).resolve()
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}-", dir=self.work_dir)).resolve()
        prepared, scale_x, scale_y = prepare_video(video_path, run_dir)
        self.runner(self.build_command(prepared, run_dir), Path(self.repo_dir))
        out = run_dir / f"{prepared.stem}_ball.csv"
        if not out.exists():
            raise RuntimeError(f"TrackNetV3 did not write {out.name} in {run_dir}")
        if prepared != video_path:
            prepared.unlink(missing_ok=True)
        df = read_ball_csv(out)
        df["x"] *= scale_x
        df["y"] *= scale_y
        return df
