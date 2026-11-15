"""Pretrained TrackNetV3 (third_party/TrackNetV3) run as a subprocess.

This is the Phase 1 stand-in tracker; the project's own trackers replace it in Phase 2.
"""
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from badminton_analyzer.data.nycu import read_ball_csv
from badminton_analyzer.tracking import tracknetv3_launcher

LAUNCHER = Path(tracknetv3_launcher.__file__)


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
        save_dir = (Path(self.work_dir) / video_path.stem).resolve()
        self.runner(self.build_command(video_path, save_dir), Path(self.repo_dir))
        out = save_dir / f"{video_path.stem}_ball.csv"
        if not out.exists():
            raise RuntimeError(f"TrackNetV3 did not write {out.name} in {save_dir}")
        return read_ball_csv(out)
