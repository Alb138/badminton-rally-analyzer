"""Score shuttle trackers on NYCU splits with the TrackNet protocol and write a results table.

Usage:
  uv run python scripts/eval_trackers.py --trackers tracknetv3 tracknet yolo \
      --splits test amateur_test --out docs/results/tracking.md

Each (tracker, split) summary is cached in outputs/eval/ and reused unless --force, so adding
a tracker later doesn't rerun the others. The tables in --out are regenerated between the
results markers; anything written outside them (analysis, notes) is kept.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from badminton_analyzer.data.nycu import index_nycu
from badminton_analyzer.data.splits import load_splits
from badminton_analyzer.tracking.evaluate import evaluate_tracker, summarize
from badminton_analyzer.tracking.factory import make_tracker

START, END = "<!-- results:start -->", "<!-- results:end -->"
TEMPLATE = f"""# Shuttle Tracking Results

Scored with the TrackNetV2/V3 protocol (`src/badminton_analyzer/tracking/evaluate.py`):
tolerance 4 px at 512 px width (10 px at 1280x720); precision = TP/(TP+FP1+FP2), recall = TP/(TP+FN).
`test` is the NYCU dataset's official test set (3 professional matches); `amateur_test` is its 3 amateur matches.
fps is wall-clock frames per second of each tracker's `track()` on this machine, startup included.

{START}
{END}

## Notes
"""


def render(summaries: list[dict], splits: list[str]) -> str:
    lines = []
    for split in splits:
        rows = sorted((s for s in summaries if s["split"] == split), key=lambda s: -s["f1"])
        lines += [f"### {split}", "",
                  "| tracker | precision | recall | F1 | accuracy | TP | FP1 | FP2 | FN | TN | frames | fps |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in rows:
            c = s["counts"]
            lines.append(f"| {s['tracker']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | "
                         f"{s['accuracy']:.3f} | {c['tp']} | {c['fp1']} | {c['fp2']} | {c['fn']} | {c['tn']} | "
                         f"{s['frames']} | {s['fps']:.1f} |")
        lines.append("")
    return "\n".join(lines)


def write_results(out: Path, block: str) -> None:
    text = out.read_text(encoding="utf-8") if out.exists() else TEMPLATE
    if START not in text or END not in text:
        text = TEMPLATE + "\n" + text
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "amateur_test"])
    parser.add_argument("--trackers-config", type=Path, default=Path("configs/trackers.yaml"))
    parser.add_argument("--nycu-root", type=Path, default=Path("data/raw/nycu"))
    parser.add_argument("--splits-file", type=Path, default=Path("configs/splits/nycu.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/tracking.md"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tracker_cfgs = yaml.safe_load(args.trackers_config.read_text(encoding="utf-8"))
    splits = load_splits(args.splits_file)
    rallies = index_nycu(args.nycu_root)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for name in args.trackers:
        tracker = None
        for split in args.splits:
            cache = args.cache_dir / f"{name}__{split}.json"
            if cache.exists() and not args.force:
                print(f"[{name}/{split}] cached", file=sys.stderr)
                continue
            tracker = tracker or make_tracker(tracker_cfgs[name])
            subset = [r for r in rallies if r.match_id in set(splits[split])]

            def progress(i, n, rally, name=name, split=split):
                print(f"[{name}/{split}] {i + 1}/{n} {rally.match_id}/{rally.rally_id}", file=sys.stderr, flush=True)

            results = evaluate_tracker(tracker, subset, progress=progress)
            summary = {"tracker": name, "split": split, **summarize(results),
                       "per_rally": [{"match": r.match_id, "rally": r.rally_id, "counts": r.counts.__dict__,
                                      "frames": r.n_frames, "seconds": r.seconds} for r in results]}
            cache.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[{name}/{split}] F1 {summary['f1']:.3f} at {summary['fps']:.1f} fps", file=sys.stderr)

    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.cache_dir.glob("*__*.json"))]
    write_results(args.out, render(summaries, args.splits))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
