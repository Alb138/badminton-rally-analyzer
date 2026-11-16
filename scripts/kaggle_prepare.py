"""Fill the Kaggle metadata for your account before pushing the dataset and kernels.

Usage: uv run python scripts/kaggle_prepare.py --username YOUR_KAGGLE_NAME [--smoke]
Writes data/cache/dataset-metadata.json (private dataset of the two training caches) and
outputs/kaggle_push/{tracknet,yolo}/ (kernel script + kernel-metadata.json) ready for
`kaggle datasets create` / `kaggle kernels push`. See docs/training.md.
"""
import argparse
import json
import sys
from pathlib import Path

DATASET = "badminton-nycu-caches"
KERNELS = {"tracknet": "tracknet_kernel.py", "yolo": "yolo_kernel.py"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", required=True)
    parser.add_argument("--smoke", action="store_true", help="push the cheap 1-epoch environment check instead")
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/kaggle_push"))
    args = parser.parse_args()

    dataset_id = f"{args.username}/{DATASET}"
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET, "id": dataset_id, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")

    for name, code_file in KERNELS.items():
        src = Path("kaggle") / name / code_file
        dst = args.out / name
        dst.mkdir(parents=True, exist_ok=True)
        code = src.read_text(encoding="utf-8")
        if args.smoke:
            code = code.replace("SMOKE = False", "SMOKE = True", 1)
        (dst / code_file).write_text(code, encoding="utf-8")
        slug = f"badminton-{name}-train" + ("-smoke" if args.smoke else "")
        (dst / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{args.username}/{slug}", "title": slug, "code_file": code_file, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [dataset_id], "competition_sources": [], "kernel_sources": [],
        }, indent=2), encoding="utf-8")
        print(f"{dst}  ->  {args.username}/{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
