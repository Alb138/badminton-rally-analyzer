import pytest

from badminton_analyzer.data.splits import load_splits, make_match_splits, save_splits

IDS = [f"m{i}" for i in range(20)]


def test_splits_are_disjoint_and_complete():
    s = make_match_splits(IDS, val_frac=0.15, test_frac=0.15, seed=42)
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(IDS)
    assert (len(s["val"]), len(s["test"])) == (3, 3)


def test_splits_ignore_input_order():
    assert make_match_splits(IDS, 0.15, 0.15, 42) == make_match_splits(IDS[::-1], 0.15, 0.15, 42)


def test_splits_change_with_seed():
    assert make_match_splits(IDS, 0.15, 0.15, 1) != make_match_splits(IDS, 0.15, 0.15, 2)


def test_splits_need_three_matches():
    with pytest.raises(ValueError):
        make_match_splits(["a", "b"], 0.15, 0.15, 42)


def test_save_load_roundtrip(tmp_path):
    s = make_match_splits(IDS, 0.15, 0.15, 42)
    save_splits(s, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == s
