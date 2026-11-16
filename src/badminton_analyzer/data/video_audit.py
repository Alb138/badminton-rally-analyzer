"""Check whether the source videos behind ShuttleSet can still be fetched."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from badminton_analyzer.schema import SchemaError


@dataclass(frozen=True)
class VideoStatus:
    match_id: str
    url: str
    available: bool
    reason: str


def load_match_urls(match_csv: Path, id_column: str, url_column: str) -> list[tuple[str, str]]:
    df = pd.read_csv(match_csv, dtype=str, keep_default_na=False)
    missing = [c for c in (id_column, url_column) if c not in df.columns]
    if missing:
        raise SchemaError(f"{match_csv} is missing columns: {', '.join(missing)}")
    return [(row[id_column], row[url_column].strip()) for _, row in df.iterrows()]


def check_urls(
    items: Sequence[tuple[str, str]], checker: Callable[[str], tuple[bool, str]]
) -> list[VideoStatus]:
    statuses = []
    for match_id, url in items:
        if not url:
            statuses.append(VideoStatus(match_id, url, False, "no url"))
            continue
        available, reason = checker(url)
        statuses.append(VideoStatus(match_id, url, available, reason))
    return statuses


def yt_dlp_checker(url: str) -> tuple[bool, str]:
    """Ask yt-dlp for the video's metadata without downloading it."""
    import yt_dlp

    options = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        return False, str(exc).strip().splitlines()[0]
    return True, ""
