"""Kaggle GPU kernel: train the TrackNet-style shuttle model (pushed with the Kaggle CLI, see docs/training.md).

Needs the private dataset with nycu_512x288/ attached and internet on. Outputs land in /kaggle/working.
"""
import glob
import json
import subprocess
import sys

REPO = "https://github.com/Alb138/badminton-rally-analyzer"
BRANCH = "phase-2"
SMOKE = False  # True: 1 epoch on 1 match, to check the Kaggle environment cheaply
EPOCHS, BATCH, WORKERS = 30, 8, 4


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 -b {BRANCH} {REPO} /kaggle/working/repo")
sh("pip install -q --no-deps --ignore-requires-python -e /kaggle/working/repo")
sh(f"{sys.executable} -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\"")

[cache] = glob.glob("/kaggle/input/*/nycu_512x288") or glob.glob("/kaggle/input/*/*/nycu_512x288")
splits = "/kaggle/working/repo/configs/splits/nycu.json"
args = f"--epochs {EPOCHS} --batch-size {BATCH}"
if SMOKE:
    splits = "/kaggle/working/smoke_splits.json"
    with open(splits, "w") as f:
        json.dump({"train": ["Professional/match1"], "val": ["Professional/match2"], "test": ["Test/match1"]}, f)
    args = f"--epochs 1 --batch-size {BATCH} --val-rallies 1"

sh(f"{sys.executable} /kaggle/working/repo/scripts/train_tracknet.py --cache-root {cache} --splits-path {splits} "
   f"--out-dir /kaggle/working/tracknet --num-workers {WORKERS} {args}")
sh("rm -rf /kaggle/working/repo")
