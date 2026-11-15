import pandas as pd
import pytest

from badminton_analyzer.data.shuttleset import (
    EXCLUDE,
    apply_class_map,
    load_shot_class_map,
    load_shuttleset_strokes,
)
from badminton_analyzer.schema import SchemaError

HEADER = "rally,ball_round,frame_num,player,type\n"


def test_load_strokes_adds_match_and_set(tmp_path):
    for m in ("matchA", "matchB"):
        (tmp_path / m).mkdir()
        (tmp_path / m / "set1.csv").write_text(HEADER + "1,1,120,A,short service\n1,2,150,B,net shot\n")
    df = load_shuttleset_strokes(tmp_path)
    assert len(df) == 4
    assert set(df["match_id"]) == {"matchA", "matchB"} and set(df["set_id"]) == {"set1"}


def test_load_strokes_names_missing_columns(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "set1.csv").write_text("rally,type\n1,smash\n")
    with pytest.raises(SchemaError, match="frame_num"):
        load_shuttleset_strokes(tmp_path)


def test_apply_class_map_merges_and_excludes():
    strokes = pd.DataFrame({"type": ["smash", "wrist smash", "unknown"]})
    out = apply_class_map(strokes, {"smash": "smash", "wrist smash": "smash", "unknown": EXCLUDE})
    assert out["shot_class"].tolist() == ["smash", "smash"]


def test_apply_class_map_rejects_unmapped_types():
    with pytest.raises(ValueError, match="lob"):
        apply_class_map(pd.DataFrame({"type": ["lob"]}), {"smash": "smash"})


def test_load_shot_class_map_rejects_unknown_class(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("smash: slam\n")
    with pytest.raises(ValueError, match="slam"):
        load_shot_class_map(p)
