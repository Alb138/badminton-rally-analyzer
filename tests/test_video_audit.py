import pytest

from badminton_analyzer.data.video_audit import check_urls, load_match_urls
from badminton_analyzer.schema import SchemaError


def test_check_urls_uses_checker():
    def fake(url):
        return (True, "") if url.endswith("ok") else (False, "Video unavailable")

    res = check_urls([("m1", "https://y/ok"), ("m2", "https://y/gone")], fake)
    assert [(r.match_id, r.available) for r in res] == [("m1", True), ("m2", False)]
    assert res[1].reason == "Video unavailable"


def test_check_urls_marks_missing_url_without_calling_checker():
    def boom(url):
        raise AssertionError("should not be called")

    assert check_urls([("m1", "")], boom)[0].reason == "no url"


def test_load_match_urls(tmp_path):
    p = tmp_path / "match.csv"
    p.write_text("video,url\nm1,https://a\nm2,\n")
    assert load_match_urls(p, "video", "url") == [("m1", "https://a"), ("m2", "")]


def test_load_match_urls_missing_column(tmp_path):
    p = tmp_path / "match.csv"
    p.write_text("video\nm1\n")
    with pytest.raises(SchemaError, match="url"):
        load_match_urls(p, "video", "url")
