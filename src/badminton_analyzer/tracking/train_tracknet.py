"""Train the TrackNet-style model on cached NYCU rallies; keep the checkpoint with the best val F1.

Runs the same way locally and on Kaggle: every path comes from TrainConfig.
"""
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from badminton_analyzer.data.splits import load_splits
from badminton_analyzer.tracking.evaluate import TOLERANCE_AT_512, Counts, count_outcomes, scores
from badminton_analyzer.tracking.tracknet_data import SEQ_LEN, RallyWindows
from badminton_analyzer.tracking.tracknet_model import TrackNetV2, decode_heatmap, wbce_loss

METRIC_COLUMNS = ["epoch", "train_loss", "val_precision", "val_recall", "val_f1", "seconds"]


@dataclass
class TrainConfig:
    cache_root: Path
    splits_path: Path
    out_dir: Path
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    num_workers: int = 2
    seed: int = 42
    amp: bool = True
    size: tuple[int, int] = (512, 288)
    val_rallies: int | None = None
    device: str | None = None


def check_disjoint(splits: dict) -> None:
    names = [n for n in ("train", "val", "test") if n in splits]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(splits[a]) & set(splits[b]))
            if shared:
                raise ValueError(f"matches in both {a} and {b}: {', '.join(shared)}")


def rally_dirs(cache_root: Path, match_ids: list[str]) -> list[Path]:
    dirs = []
    for match in sorted(match_ids):
        match_dir = Path(cache_root) / match
        if not match_dir.is_dir():
            raise FileNotFoundError(f"no cached frames for {match} under {cache_root}")
        dirs += sorted(d for d in match_dir.iterdir() if (d / "meta.json").exists())
    return dirs


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_windows(model: torch.nn.Module, dirs: list[Path], size, device: str, batch_size: int) -> Counts:
    """Score the model on non-overlapping windows, in input space (tolerance 4 px at 512 width)."""
    model.eval()
    loader = DataLoader(RallyWindows(dirs, size=size, stride=SEQ_LEN), batch_size=batch_size)
    gt_rows, pred_rows, frame = [], [], 0
    for x, _, coords in loader:
        heatmaps = model(x.to(device)).float().cpu().numpy()
        for window, window_coords in zip(heatmaps, coords.numpy()):
            for hm, (vis, cx, cy) in zip(window, window_coords):
                visible, px, py = decode_heatmap(hm)
                gt_rows.append((frame, vis > 0, cx, cy))
                pred_rows.append((frame, visible, px, py))
                frame += 1
    columns = ["frame", "visible", "x", "y"]
    tolerance = TOLERANCE_AT_512 * size[0] / 512
    return count_outcomes(pd.DataFrame(gt_rows, columns=columns), pd.DataFrame(pred_rows, columns=columns), tolerance)


def train(cfg: TrainConfig) -> Path:
    splits = load_splits(cfg.splits_path)
    check_disjoint(splits)
    _seed_everything(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.startswith("cuda")
    size = tuple(cfg.size)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    config["size"] = list(size)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = RallyWindows(rally_dirs(cfg.cache_root, splits["train"]), size=size, stride=1, augment=True,
                            seed=cfg.seed)
    val_dirs = rally_dirs(cfg.cache_root, splits["val"])[: cfg.val_rallies]
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        generator=torch.Generator().manual_seed(cfg.seed), pin_memory=device.startswith("cuda"),
                        persistent_workers=cfg.num_workers > 0)
    model = TrackNetV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [max(1, round(cfg.epochs * 2 / 3))], gamma=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(METRIC_COLUMNS)
    best_f1 = -1.0
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                pred = model(x)
            loss = wbce_loss(pred.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(x)
            seen += len(x)
        scheduler.step()
        val = scores(evaluate_windows(model, val_dirs, size, device, cfg.batch_size))
        seconds = time.perf_counter() - start
        with open(out / "metrics.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, total / max(seen, 1), val["precision"], val["recall"], val["f1"], seconds])
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "val_f1": val["f1"], "size": list(size)}
        torch.save(checkpoint, out / "last.pt")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(checkpoint, out / "best.pt")
        print(f"epoch {epoch}/{cfg.epochs} loss {total / max(seen, 1):.5f} val F1 {val['f1']:.3f} "
              f"(P {val['precision']:.3f} R {val['recall']:.3f}) {seconds:.0f}s", file=sys.stderr, flush=True)
    return out / "best.pt"
