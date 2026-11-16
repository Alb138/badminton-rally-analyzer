"""Decode each rally once into resized JPEG frames, so training doesn't re-decode video every epoch."""
import json
import shutil
from pathlib import Path

import cv2

from badminton_analyzer.video_io import iter_frames, read_video_meta

INPUT_SIZE = (512, 288)  # (w, h) TrackNet's input


def cache_rally(video_path: Path, label_path: Path, out_dir: Path, size=INPUT_SIZE, quality: int = 95) -> int:
    """Write ``out_dir/00000.jpg ...``, ``labels.csv`` and ``meta.json``; return the number of frames.

    A cache that's already complete for this video and size is reused as is.
    """
    out_dir = Path(out_dir)
    src = read_video_meta(Path(video_path))
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        complete = len(list(out_dir.glob("*.jpg"))) == meta["n_frames"]
        if complete and meta["source_frame_count"] == src.n_frames and meta["size"] == list(size):
            return meta["n_frames"]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for n, frame in enumerate(iter_frames(Path(video_path)), start=1):
        small = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{n - 1:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    shutil.copyfile(label_path, out_dir / "labels.csv")
    meta = {"width": src.width, "height": src.height, "fps": src.fps, "n_frames": n,
            "source_frame_count": src.n_frames, "size": list(size)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n
