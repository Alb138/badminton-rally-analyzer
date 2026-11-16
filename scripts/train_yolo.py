"""Fine-tune YOLOv8 as a single-class shuttle detector on the dataset from build_yolo_dataset.py.

Usage:
  uv run python scripts/train_yolo.py --data data/cache/yolo_nycu/data.yaml --epochs 30 --imgsz 1280 --name yolov8s
Copies the best weights to <out>/<name>/best.pt and prints that path.
"""
import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=float, default=-1, help="-1 = auto batch size (fills ~60%% of GPU memory)")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of training images to use")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/runs/yolo"))
    parser.add_argument("--name", default="yolov8s")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    YOLO(args.model).train(data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=batch,
                           fraction=args.fraction, workers=args.workers, project=str(args.out.resolve()),
                           name=args.name, exist_ok=True, seed=42, deterministic=True, single_cls=True,
                           device=args.device)
    run_dir = args.out / args.name
    best = run_dir / "best.pt"
    shutil.copyfile(run_dir / "weights" / "best.pt", best)
    print(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
