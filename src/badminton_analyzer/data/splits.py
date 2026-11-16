"""Match-level train/val/test splits: no match ever appears in two splits."""
import json
import random
from collections.abc import Sequence
from pathlib import Path


def make_match_splits(
    match_ids: Sequence[str], val_frac: float, test_frac: float, seed: int
) -> dict[str, list[str]]:
    ids = sorted(set(match_ids))
    n = len(ids)
    if n < 3:
        raise ValueError(f"need at least 3 matches to split, got {n}")
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0  # 0: dataset has its own test set
    n_val = max(1, round(n * val_frac))
    if n_test + n_val >= n:
        raise ValueError(f"val/test fractions leave no training matches out of {n}")
    return {
        "train": sorted(ids[n_test + n_val :]),
        "val": sorted(ids[n_test : n_test + n_val]),
        "test": sorted(ids[:n_test]),
    }


class SplitMismatchError(RuntimeError):
    """The splits computed from the data on disk differ from the committed split file."""


def sync_splits(splits: dict, path: Path, write: bool) -> None:
    """Write ``splits`` only when asked; otherwise check them against the committed file.

    The committed file is the source of truth: a re-download, a different unzip folder or a
    video taken down must never silently move matches between train and test.
    """
    path = Path(path)
    if write:
        save_splits(splits, path)
        return
    if not path.exists():
        raise SplitMismatchError(f"{path.as_posix()} does not exist; create it with --write-splits")
    if load_splits(path) != splits:
        raise SplitMismatchError(
            f"{path.as_posix()} differs from the splits computed from the data on disk; "
            "the committed file is kept. Rerun with --write-splits only if the change is intended."
        )


def save_splits(splits: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


def load_splits(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
