import pytest

from badminton_analyzer.data.splits import (
    SplitMismatchError,
    load_splits,
    make_match_splits,
    save_splits,
    sync_splits,
)

IDS = [f"m{i}" for i in range(20)]


def test_splits_are_disjoint_and_complete():
    s = make_match_splits(IDS, val_frac=0.15, test_frac=0.15, seed=42)
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(IDS)
    assert (len(s["val"]), len(s["test"])) == (3, 3)


def test_splits_ignore_input_order():
    assert make_match_splits(IDS, 0.15, 0.15, 42) == make_match_splits(IDS[::-1], 0.15, 0.15, 42)


def test_splits_change_with_seed():
    assert make_match_splits(IDS, 0.15, 0.15, 1) != make_match_splits(IDS, 0.15, 0.15, 2)


def test_zero_test_fraction_leaves_test_empty():
    # Used when a dataset ships its own official test matches.
    s = make_match_splits(IDS, val_frac=0.15, test_frac=0.0, seed=42)
    assert s["test"] == [] and len(s["val"]) == 3 and len(s["train"]) == 17


def test_splits_need_three_matches():
    with pytest.raises(ValueError):
        make_match_splits(["a", "b"], 0.15, 0.15, 42)


def test_save_load_roundtrip(tmp_path):
    s = make_match_splits(IDS, 0.15, 0.15, 42)
    save_splits(s, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == s


def test_sync_splits_writes_only_when_asked(tmp_path):
    s = make_match_splits(IDS, 0.15, 0.15, 42)
    with pytest.raises(SplitMismatchError, match="--write-splits"):
        sync_splits(s, tmp_path / "s.json", write=False)
    sync_splits(s, tmp_path / "s.json", write=True)
    assert load_splits(tmp_path / "s.json") == s


def test_sync_splits_accepts_matching_committed_file(tmp_path):
    s = make_match_splits(IDS, 0.15, 0.15, 42)
    save_splits(s, tmp_path / "s.json")
    sync_splits(s, tmp_path / "s.json", write=False)


def test_sync_splits_refuses_to_overwrite_a_different_file(tmp_path):
    # e.g. a YouTube takedown changes the available-video set: the committed split stays the truth.
    save_splits(make_match_splits(IDS, 0.15, 0.15, 42), tmp_path / "s.json")
    with pytest.raises(SplitMismatchError, match="differs"):
        sync_splits(make_match_splits(IDS[:-1], 0.15, 0.15, 42), tmp_path / "s.json", write=False)
    assert load_splits(tmp_path / "s.json") == make_match_splits(IDS, 0.15, 0.15, 42)
