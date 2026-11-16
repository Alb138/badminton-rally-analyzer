"""Phase 0 data audit: dataset summaries, ShuttleSet video availability and match-level splits.

Usage: uv run python scripts/audit_data.py [--config configs/data.yaml] [--out docs/data_audit.md] [--write-splits]

The committed configs/splits/*.json files are the source of truth. Without --write-splits the
script only checks that the data on disk still gives the same splits, and exits with 1 if not.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from badminton_analyzer.data.nycu import index_nycu, read_ball_csv
from badminton_analyzer.data.shuttleset import apply_class_map, load_shot_class_map, load_shuttleset_strokes
from badminton_analyzer.data.splits import SplitMismatchError, make_match_splits, sync_splits
from badminton_analyzer.data.video_audit import check_urls, load_match_urls, yt_dlp_checker

SPLITS_DIR = Path("configs/splits")


def _table(df: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join(" --- " for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return [header, rule, *rows, ""]


def _sync(splits: dict, name: str, write: bool, problems: list[str]) -> list[str]:
    path = SPLITS_DIR / name
    summary = ", ".join(f"{k} {len(v)}" for k, v in splits.items())
    try:
        sync_splits(splits, path, write)
    except SplitMismatchError as exc:
        problems.append(str(exc))
        return [f"**WARNING:** {exc}", ""]
    verb = "written to" if write else "match the committed"
    return [f"Splits {verb} `{path.as_posix()}`: {summary}", ""]


def nycu_section(cfg: dict, split_cfg: dict, write: bool, problems: list[str]) -> list[str]:
    lines = ["## NYCU Shuttlecock Trajectory Dataset", ""]
    root = Path(cfg["root"])
    rallies = index_nycu(root) if root.exists() else []
    if not rallies:
        return lines + [f"Not downloaded yet: no rallies found under `{root.as_posix()}`.", ""]
    rows = []
    for r in rallies:
        labels = read_ball_csv(r.label_path)
        rows.append({"match": r.match_id, "frames": len(labels), "visible": int(labels["visible"].sum())})
    per_rally = pd.DataFrame(rows)
    per_match = per_rally.groupby("match").agg(
        rallies=("frames", "size"), frames=("frames", "sum"), visible=("visible", "sum")
    ).reset_index()
    per_match["visible_share"] = (per_match["visible"] / per_match["frames"]).round(3)
    test_pattern = cfg.get("test_match_pattern")
    per_match["group"] = "professional"
    if test_pattern:
        per_match.loc[per_match["match"].str.contains(test_pattern, regex=False), "group"] = "official test"
    per_match.loc[per_match["match"].str.contains(cfg["amateur_match_pattern"], regex=False), "group"] = "amateur"
    by_group = {g: sorted(per_match.loc[per_match["group"] == g, "match"]) for g in ("professional", "official test", "amateur")}
    lines += [
        f"Matches: {len(per_match)} ({len(by_group['professional'])} professional, "
        f"{len(by_group['official test'])} official test, {len(by_group['amateur'])} amateur). "
        f"Rallies: {len(per_rally)}. Frames: {int(per_match['frames'].sum())}.",
        "",
        *_table(per_match.drop(columns="visible")),
    ]
    # The dataset's own test matches (the ones TrackNetV2/V3 report on) become our test split,
    # so tracking results are comparable with published numbers.
    test_frac = 0.0 if by_group["official test"] else split_cfg["test_frac"]
    splits = make_match_splits(by_group["professional"], split_cfg["val_frac"], test_frac, split_cfg["seed"])
    if by_group["official test"]:
        splits["test"] = by_group["official test"]
    splits["amateur_test"] = by_group["amateur"]
    return lines + _sync(splits, "nycu.json", write, problems)


def shuttleset_section(cfg: dict, split_cfg: dict, write: bool, problems: list[str]) -> list[str]:
    lines = ["## ShuttleSet", ""]
    strokes = load_shuttleset_strokes(Path(cfg["root"]))
    lines += [f"Matches: {strokes['match_id'].nunique()}. Strokes: {len(strokes)}.", "", "### Raw shot types", ""]
    raw = strokes["type"].value_counts().rename_axis("type").reset_index(name="strokes")
    lines += _table(raw)
    try:
        merged = apply_class_map(strokes, load_shot_class_map(Path(cfg["class_map"])))
    except (ValueError, FileNotFoundError) as exc:
        lines += [f"Merged classes unavailable: {exc}", ""]
    else:
        counts = merged["shot_class"].value_counts().rename_axis("class").reset_index(name="strokes")
        counts["under_500"] = counts["strokes"] < 500
        lines += ["### Merged shot classes", "", f"Strokes kept: {len(merged)}.", "", *_table(counts)]

    statuses = check_urls(
        load_match_urls(Path(cfg["match_csv"]), cfg["match_id_column"], cfg["url_column"]), yt_dlp_checker
    )
    available = [s.match_id for s in statuses if s.available]
    status_df = pd.DataFrame(
        [{"match": s.match_id, "available": "yes" if s.available else "no", "reason": s.reason} for s in statuses]
    )
    lines += ["### Source video availability", "", f"Available videos: {len(available)}/{len(statuses)}", "",
              *_table(status_df)]
    if len(available) >= 3:
        splits = make_match_splits(available, split_cfg["val_frac"], split_cfg["test_frac"], split_cfg["seed"])
        lines += ["Splits are over the matches whose video is available.", ""]
        lines += _sync(splits, "shuttleset.json", write, problems)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, default=Path("docs/data_audit.md"))
    parser.add_argument("--write-splits", action="store_true", help="(re)write configs/splits/*.json")
    args = parser.parse_args()
    problems: list[str] = []
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    lines = ["# Phase 0 Data Audit", "", "Generated by `scripts/audit_data.py`.", ""]
    lines += nycu_section(cfg["nycu"], cfg["splits"], args.write_splits, problems)
    lines += shuttleset_section(cfg["shuttleset"], cfg["splits"], args.write_splits, problems)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
