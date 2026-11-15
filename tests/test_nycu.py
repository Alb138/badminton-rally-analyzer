import pytest

from badminton_analyzer.data.nycu import index_nycu, read_ball_csv


def test_read_ball_csv_sets_invisible_to_nan(tmp_path):
    p = tmp_path / "1_01_00_ball.csv"
    p.write_text("Frame,Visibility,X,Y\n0,1,100,50\n1,0,0,0\n")
    df = read_ball_csv(p)
    assert list(df.columns) == ["frame", "visible", "x", "y"]
    assert df["visible"].tolist() == [True, False]
    assert df.loc[1, ["x", "y"]].isna().all()


def test_index_nycu_pairs_labels_with_videos(tmp_path):
    for m in ("match1", "match2"):
        (tmp_path / "train" / m / "csv").mkdir(parents=True)
        (tmp_path / "train" / m / "video").mkdir(parents=True)
        (tmp_path / "train" / m / "csv" / "1_01_00_ball.csv").write_text("Frame,Visibility,X,Y\n")
    (tmp_path / "train" / "match1" / "video" / "1_01_00.mp4").write_bytes(b"")
    with pytest.warns(UserWarning, match="1 label file"):
        rallies = index_nycu(tmp_path)
    assert [(r.match_id, r.rally_id) for r in rallies] == [("train/match1", "1_01_00")]


def test_index_nycu_reads_published_layout(tmp_path):
    for top in ("profession_dataset", "amateur_dataset"):
        match = tmp_path / "TrackNetV2" / top / "match1"
        (match / "ball_trajectory").mkdir(parents=True)
        (match / "rally_video").mkdir(parents=True)
        (match / "ball_trajectory" / "1_04_02_ball.csv").write_text("Frame,Visibility,X,Y\n")
        (match / "rally_video" / "1_04_02.mp4").write_bytes(b"")
    rallies = index_nycu(tmp_path)
    assert [r.match_id for r in rallies] == [
        "TrackNetV2/amateur_dataset/match1",
        "TrackNetV2/profession_dataset/match1",
    ]
    assert rallies[0].video_path.name == "1_04_02.mp4"
