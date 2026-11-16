"""Kaggle GPU kernel: fine-tune YOLOv8 as a shuttle detector (pushed with the Kaggle CLI, see docs/training.md).

Needs the private dataset with yolo_nycu/ attached and internet on. Outputs land in /kaggle/working.
"""
import glob
import subprocess
import sys

REPO = "https://github.com/Alb138/badminton-rally-analyzer"
BRANCH = "phase-2"
SMOKE = False  # True: 1 epoch on 2% of the images, to check the Kaggle environment cheaply
EPOCHS, IMGSZ, WORKERS = 30, 1280, 4


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 -b {BRANCH} {REPO} /kaggle/working/repo")
sh("pip install -q --no-deps --ignore-requires-python -e /kaggle/working/repo")
sh("pip install -q ultralytics")

[data_yaml] = glob.glob("/kaggle/input/*/yolo_nycu/data.yaml") or glob.glob("/kaggle/input/*/*/yolo_nycu/data.yaml")
extra = "--epochs 1 --fraction 0.02" if SMOKE else f"--epochs {EPOCHS}"
sh(f"{sys.executable} /kaggle/working/repo/scripts/train_yolo.py --data {data_yaml} --imgsz {IMGSZ} --batch -1 "
   f"--workers {WORKERS} --out /kaggle/working/yolo --name yolov8s {extra}")
sh("rm -rf /kaggle/working/repo")
