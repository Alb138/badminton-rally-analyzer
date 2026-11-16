"""Build the YOLO shuttle-detection dataset (train and val matches only) from NYCU point labels.

Usage: uv run python scripts/build_yolo_dataset.py --nycu-root data/raw/nycu \
           --splits configs/splits/nycu.json --out data/cache/yolo_nycu --stride 3
"""
import argparse
import sys
from pathlib import Path

from badminton_analyzer.data.nycu import index_nycu
from badminton_analyzer.data.splits import load_splits
from badminton_analyzer.tracking.yolo_data import build_yolo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nycu-root", type=Path, default=Path("data/raw/nycu"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits/nycu.json"))
    parser.add_argument("--out", type=Path, default=Path("data/cache/yolo_nycu"))
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    splits = load_splits(args.splits)
    rallies = index_nycu(args.nycu_root)
    by_split = {name: [r for r in rallies if r.match_id in set(splits[name])] for name in ("train", "val")}
    data_yaml = build_yolo_dataset(by_split, args.out, stride=args.stride)
    for name in by_split:
        n = len(list((args.out / "images" / name).glob("*.jpg")))
        print(f"{name}: {len(by_split[name])} rallies, {n} images", file=sys.stderr)
    print(data_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
