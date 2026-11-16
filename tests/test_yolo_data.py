from badminton_analyzer.data.nycu import NycuRally
from badminton_analyzer.tracking.yolo_data import build_yolo_dataset, yolo_label_line
from tests.helpers import make_video


def test_label_line_is_normalized():
    assert yolo_label_line(640, 360, 1280, 720) == "0 0.500000 0.500000 0.012500 0.022222"


def test_label_box_is_clipped_at_the_border():
    cx, cy, w, h = map(float, yolo_label_line(2, 2, 1280, 720).split()[1:])
    assert cx - w / 2 >= 0 and cy - h / 2 >= 0


def test_build_dataset_strides_and_keeps_negatives(tmp_path):
    # one 7-frame 128x64 rally in train, shuttle at (40, 20), invisible on frame 3
    video = make_video(tmp_path / "v.mp4", n_frames=7, size=(128, 64))
    labels = tmp_path / "1_01_00_ball.csv"
    labels.write_text("Frame,Visibility,X,Y\n" + "".join(
        f"{i},{0 if i == 3 else 1},{0 if i == 3 else 40},{0 if i == 3 else 20}\n" for i in range(7)))
    rally = NycuRally("Professional/match1", "1_01_00", video, labels)
    data_yaml = build_yolo_dataset({"train": [rally], "val": []}, tmp_path / "y", stride=3)
    imgs = sorted((tmp_path / "y" / "images" / "train").glob("*.jpg"))
    assert [p.stem.split("__")[-1] for p in imgs] == ["00000", "00003", "00006"]
    assert (tmp_path / "y" / "labels" / "train" / (imgs[1].stem + ".txt")).read_text() == ""
    assert (tmp_path / "y" / "labels" / "train" / (imgs[0].stem + ".txt")).read_text().startswith("0 0.312500 0.312500")
    assert "names" in data_yaml.read_text()
