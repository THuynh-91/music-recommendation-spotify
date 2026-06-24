from __future__ import annotations

"""Real, no-auth recommender used in demo mode.

Pipeline (no user credentials, no Spotify app required):
  1. Resolve the pasted Spotify track URL -> real title via Spotify's PUBLIC
     oEmbed endpoint (https://open.spotify.com/oembed).
  2. Resolve title -> {artist, deezer artist id} via Deezer's public search API.
  3. Pull REAL similar tracks from Deezer: related artists' top tracks plus the
     seed artist's own top tracks. Every item returned is a real Deezer track.

Spotify deprecated audio-features / recommendations (Nov 2024), so there is no
"audio feature" similarity here and we do not pretend there is. Relevance is
Deezer's editorial "related artists" graph, which is genuine collaborative data.

This is a direct port of ``lib/recommender.ts`` so the demo backend serves the
SAME real data the credentialed (non-demo) path would.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

from ..schemas.recommend import RecommendResponse, RecommendationItem, SeedTrack
from ..spotify.parsing import SpotifyEntity

logger = logging.getLogger("realtime_recommender")

OEMBED_ENDPOINT = "https://open.spotify.com/oembed"
DEEZER_API = "https://api.deezer.com"
_USER_AGENT = "spotify-rec/1.0 (+https://localhost)"


class RecommenderError(Exception):
    """Raised when the real pipeline cannot resolve real recommendations."""


async def _fetch_json(client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    res = await client.get(url, params=params)
    if res.status_code >= 400:
        raise RecommenderError(f"Upstream request failed ({res.status_code}) for {res.request.url}")
    return res.json()


def _spotify_search_link(title: str, artist: str) -> str:
    return f"https://open.spotify.com/search/{quote(f'{title} {artist}')}"


async def _resolve_seed(client: httpx.AsyncClient, url: str) -> Tuple[str, str, int, Optional[str]]:
    """Resolve a Spotify track URL to (title, artist, deezer_artist_id, image)."""
    try:
        oembed = await _fetch_json(client, OEMBED_ENDPOINT, {"url": url})
    except RecommenderError as exc:
        raise RecommenderError(
            f"Could not read the Spotify track via oEmbed (is the URL a public track?): {exc}"
        ) from exc

    query = (oembed.get("title") or "").strip()
    artist_hint = (oembed.get("author_name") or "").strip() or None
    image = oembed.get("thumbnail_url") or None
    if not query:
        raise RecommenderError("Spotify oEmbed returned no title for this URL.")

    search_query = f"{query} {artist_hint}" if artist_hint else query
    search = await _fetch_json(client, f"{DEEZER_API}/search", {"q": search_query, "limit": 5})
    best = _first_with_artist(search.get("data"))
    if best is None:
        # Retry with the bare title in case the artist hint hurt the match.
        retry = await _fetch_json(client, f"{DEEZER_API}/search", {"q": query, "limit": 5})
        best = _first_with_artist(retry.get("data"))
    if best is None:
        raise RecommenderError(
            f'Resolved Spotify title "{query}" but could not match it to an artist on Deezer.'
        )

    artist = best.get("artist") or {}
    return (
        best.get("title") or query,
        artist.get("name") or artist_hint or "Unknown artist",
        int(artist["id"]),
        image,
    )


def _first_with_artist(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, list):
        return None
    for track in data:
        if isinstance(track, dict) and (track.get("artist") or {}).get("id"):
            return track
    return None


async def build_real_recommendations(url: str, limit: int) -> RecommendResponse:
    """Build EXACTLY ``limit`` real recommendations from Deezer related artists."""
    target = min(max(int(limit), 1), 50)

    async with httpx.AsyncClient(
        timeout=20.0,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        seed_title, seed_artist, artist_id, seed_image = await _resolve_seed(client, url)

        related = await _fetch_json(client, f"{DEEZER_API}/artist/{artist_id}/related", {"limit": 25})
        related_artists = [a for a in (related.get("data") or []) if isinstance(a, dict) and a.get("id")]

        seen: set[str] = set()
        items: List[RecommendationItem] = []

        def add_track(track: Dict[str, Any], reason: str) -> None:
            name = track.get("title") or track.get("title_short")
            artist_obj = track.get("artist") or {}
            artist = artist_obj.get("name")
            if not name or not artist:
                return
            key = f"{name.lower()}::{artist.lower()}"
            if key in seen:
                return
            # Skip the exact seed track itself.
            if name.lower() == seed_title.lower() and artist.lower() == seed_artist.lower():
                return
            seen.add(key)
            items.append(
                RecommendationItem(
                    track_id=f"dz-{len(items)}",
                    name=name,
                    artists=[artist],
                    preview_url=track.get("preview") or None,
                    external_url=track.get("link") or _spotify_search_link(name, artist),
                    image_url=None,
                    similarity=0.0,
                    explanation=reason,
                )
            )

        # Pass 1: 2 tracks per related artist (breadth). Pass 2: fill from remaining.
        for pass_limit in (2, 5):
            for artist in related_artists:
                if len(items) >= target:
                    break
                try:
                    top = await _fetch_json(
                        client, f"{DEEZER_API}/artist/{artist['id']}/top", {"limit": pass_limit}
                    )
                except RecommenderError:
                    continue
                for track in top.get("data") or []:
                    if len(items) >= target:
                        break
                    add_track(track, f"Related to {seed_artist} (via {artist.get('name')})")
            if len(items) >= target:
                break

        # Fallback: top up with the seed artist's own catalogue if related pool was thin.
        if len(items) < target:
            try:
                own_top = await _fetch_json(
                    client, f"{DEEZER_API}/artist/{artist_id}/top", {"limit": 50}
                )
                for track in own_top.get("data") or []:
                    if len(items) >= target:
                        break
                    add_track(track, f"More from {seed_artist}")
            except RecommenderError:
                pass

    return RecommendResponse(
        type="track",
        seed_track=SeedTrack(
            id=f"dz-seed-{artist_id}",
            name=seed_title,
            artists=[seed_artist],
            image_url=seed_image,
            external_url=url,
        ),
        recommendations=items[:target],
    )


def canonical_track_url(entity: SpotifyEntity) -> str:
    """Reconstruct a public Spotify URL from a parsed entity for oEmbed lookups."""
    return f"https://open.spotify.com/{entity.kind}/{entity.id}"
