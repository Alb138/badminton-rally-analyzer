"""Train the TrackNet-style shuttle model on cached NYCU rallies.

Usage:
  uv run python scripts/train_tracknet.py --cache-root data/cache/nycu_512x288 \
      --splits-path configs/splits/nycu.json --out-dir outputs/runs/tracknet --epochs 30 --batch-size 8
"""
import argparse
import sys
from pathlib import Path

from badminton_analyzer.tracking.train_tracknet import TrainConfig, train


def main() -> int:
    d = TrainConfig(Path("."), Path("."), Path("."))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--splits-path", type=Path, default=Path("configs/splits/nycu.json"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    p.add_argument("--width", type=int, default=d.size[0])
    p.add_argument("--height", type=int, default=d.size[1])
    p.add_argument("--val-rallies", type=int, default=None, help="score only the first N val rallies each epoch")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    best = train(TrainConfig(a.cache_root, a.splits_path, a.out_dir, epochs=a.epochs, batch_size=a.batch_size,
                             lr=a.lr, num_workers=a.num_workers, seed=a.seed, amp=not a.no_amp,
                             size=(a.width, a.height), val_rallies=a.val_rallies, device=a.device))
    print(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
