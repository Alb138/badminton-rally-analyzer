import pytest

from badminton_analyzer.schema import (
    FRAME_COLUMNS,
    KPT_LEN,
    SHOT_COLUMNS,
    SchemaError,
    empty_frames_table,
    empty_shots_table,
    validate_frames,
    validate_shots,
)


def test_empty_frames_table_is_valid():
    df = empty_frames_table(5)
    validate_frames(df)
    assert list(df.columns) == FRAME_COLUMNS
    assert df["frame"].tolist() == [0, 1, 2, 3, 4]
    assert not df["shuttle_visible"].any() and not df["near_detected"].any()
    assert len(df.loc[0, "near_kpts"]) == KPT_LEN


def test_validate_frames_names_missing_column():
    with pytest.raises(SchemaError, match="far_court_y"):
        validate_frames(empty_frames_table(3).drop(columns=["far_court_y"]))


def test_validate_frames_rejects_non_contiguous_frames():
    with pytest.raises(SchemaError, match="contiguous"):
        validate_frames(empty_frames_table(3).drop(index=1))


def test_empty_shots_table_is_valid():
    df = empty_shots_table()
    validate_shots(df)
    assert list(df.columns) == SHOT_COLUMNS and len(df) == 0
