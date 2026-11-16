"""Cache every NYCU rally as 512x288 JPEG frames for TrackNet-style training.

Usage: uv run python scripts/build_frame_cache.py --nycu-root data/raw/nycu --out data/cache/nycu_512x288
Layout: <out>/<match_id>/<rally_id>/{00000.jpg, ..., labels.csv, meta.json}
"""
import argparse
import sys
from pathlib import Path

from badminton_analyzer.data.nycu import index_nycu
from badminton_analyzer.tracking.frames_cache import INPUT_SIZE, cache_rally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nycu-root", type=Path, default=Path("data/raw/nycu"))
    parser.add_argument("--out", type=Path, default=Path("data/cache/nycu_512x288"))
    parser.add_argument("--width", type=int, default=INPUT_SIZE[0])
    parser.add_argument("--height", type=int, default=INPUT_SIZE[1])
    args = parser.parse_args()
    rallies = index_nycu(args.nycu_root)
    total = 0
    for i, r in enumerate(rallies, start=1):
        n = cache_rally(r.video_path, r.label_path, args.out / r.match_id / r.rally_id, size=(args.width, args.height))
        total += n
        print(f"{i}/{len(rallies)} {r.match_id}/{r.rally_id}: {n} frames", file=sys.stderr, flush=True)
    print(f"cached {len(rallies)} rallies, {total} frames under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
