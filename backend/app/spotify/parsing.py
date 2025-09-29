from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SPOTIFY_URL_RE = re.compile(
    r"https?://(?:open|play)\.spotify\.com/(?P<type>track|playlist)/(?P<id>[A-Za-z0-9]{22})",
    re.IGNORECASE,
)
SPOTIFY_URI_RE = re.compile(r"spotify:(?P<type>track|playlist):(?P<id>[A-Za-z0-9]{22})", re.IGNORECASE)


@dataclass(slots=True)
class SpotifyEntity:
    kind: Literal["track", "playlist"]
    id: str


def parse_spotify_url(value: str) -> SpotifyEntity:
    value = value.strip()
    m = SPOTIFY_URI_RE.match(value)
    if not m:
        m = SPOTIFY_URL_RE.search(value)
    if not m:
        raise ValueError("unsupported spotify url")
    kind = m.group("type").lower()
    spotify_id = m.group("id")
    return SpotifyEntity(kind=kind, id=spotify_id)