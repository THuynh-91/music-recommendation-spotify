"""Tests for the demo-mode recommendation builder.

DEMO mode is the only mode that boots without infra, so it must return REAL songs
via the public Spotify oEmbed -> Deezer pipeline (no credentials). These tests
exercise that pipeline with a mocked httpx transport so they assert the real code
path without hitting the network.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import realtime_recommender as rr
from app.services.realtime_recommender import (
    RecommenderError,
    build_real_recommendations,
    canonical_track_url,
)
from app.spotify.parsing import SpotifyEntity

# Capture the genuine httpx.AsyncClient before any test patches it.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _related(n: int) -> dict:
    return {"data": [{"id": 1000 + i, "name": f"Artist {i}"} for i in range(n)]}


def _top(artist_idx: int, count: int) -> dict:
    return {
        "data": [
            {
                "title": f"Song {artist_idx}-{j}",
                "artist": {"name": f"Artist {artist_idx}"},
                "link": f"https://www.deezer.com/track/{artist_idx}{j}",
                "preview": None,
            }
            for j in range(count)
        ]
    }


def _make_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "open.spotify.com/oembed" in url:
            return httpx.Response(200, json={"title": "Candy Paint", "author_name": "Post Malone", "thumbnail_url": "https://i.example/x.jpg"})
        if "api.deezer.com/search" in url:
            return httpx.Response(200, json={"data": [{"title": "Candy Paint", "artist": {"id": 7543848, "name": "Post Malone"}}]})
        if "/related" in url:
            return httpx.Response(200, json=_related(25))
        if "/top" in url:
            # extract artist id from /artist/<id>/top
            parts = request.url.path.split("/")
            artist_id = parts[parts.index("artist") + 1]
            return httpx.Response(200, json=_top(int(artist_id) % 1000, 5))
        return httpx.Response(404, json={})

    return handler


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    handler = _make_handler()

    class _PatchedClient(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rr.httpx, "AsyncClient", _PatchedClient)


def test_canonical_track_url() -> None:
    entity = SpotifyEntity(kind="track", id="32lItqlMi4LBhb4k0BaSaC")
    assert canonical_track_url(entity) == "https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC"


def test_returns_real_seed_and_exact_limit() -> None:
    resp = asyncio.run(
        build_real_recommendations("https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC", 15)
    )
    assert resp.type == "track"
    assert resp.seed_track is not None
    # Real seed resolved via oEmbed -> Deezer, NOT a hardcoded mock.
    assert resp.seed_track.name == "Candy Paint"
    assert resp.seed_track.artists == ["Post Malone"]
    assert len(resp.recommendations) == 15
    # No fabricated/demo markers anywhere.
    for item in resp.recommendations:
        assert "demo data" not in item.explanation.lower()
        assert item.name
        assert item.artists
        assert item.external_url
        assert 0.0 <= item.similarity <= 1.0


def test_no_duplicate_tracks() -> None:
    resp = asyncio.run(
        build_real_recommendations("https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC", 20)
    )
    keys = {(r.name.lower(), r.artists[0].lower()) for r in resp.recommendations}
    assert len(keys) == len(resp.recommendations)


def test_oembed_failure_raises_recommender_error(monkeypatch) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    class _Failing(_REAL_ASYNC_CLIENT):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(failing)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rr.httpx, "AsyncClient", _Failing)
    with pytest.raises(RecommenderError):
        asyncio.run(
            build_real_recommendations("https://open.spotify.com/track/32lItqlMi4LBhb4k0BaSaC", 5)
        )
