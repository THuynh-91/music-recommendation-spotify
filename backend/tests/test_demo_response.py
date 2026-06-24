"""Tests for the demo-mode recommendation builder.

These cover the deterministic mock path that the no-infra DEMO deployment relies
on, so the demo experience stays correct even if the live pipeline changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.recommendations import build_demo_response
from app.spotify.parsing import SpotifyEntity


def test_demo_track_response_shape() -> None:
    entity = SpotifyEntity(kind="track", id="2TpxZ7JUBn3uw46aR7qd6V")
    resp = build_demo_response(entity, limit=5)

    assert resp.type == "track"
    assert resp.seed_track is not None
    assert resp.seed_track.id == entity.id
    assert resp.seed_playlist is None
    assert len(resp.recommendations) == 5

    first = resp.recommendations[0]
    assert first.name
    assert first.artists
    assert "demo data" in first.explanation
    # Similarity must be a sane, descending-ish ranked score in [0, 1].
    assert 0.0 <= first.similarity <= 1.0


def test_demo_playlist_response_shape() -> None:
    entity = SpotifyEntity(kind="playlist", id="37i9dQZF1DXcBWIGoYBM5M")
    resp = build_demo_response(entity, limit=3)

    assert resp.type == "playlist"
    assert resp.seed_playlist is not None
    assert resp.seed_playlist.id == entity.id
    assert resp.playlist is not None
    assert resp.playlist.ingested_tracks == 25
    assert len(resp.recommendations) == 3


def test_demo_limit_is_clamped_to_available_tracks() -> None:
    entity = SpotifyEntity(kind="track", id="2TpxZ7JUBn3uw46aR7qd6V")
    # Asking for more than the fixture supplies should clamp, not error.
    resp = build_demo_response(entity, limit=999)
    assert 1 <= len(resp.recommendations) <= 8

    # A floor of at least one recommendation even for a tiny/zero limit.
    resp_min = build_demo_response(entity, limit=0)
    assert len(resp_min.recommendations) >= 1


def test_demo_recommendations_are_deterministic() -> None:
    entity = SpotifyEntity(kind="track", id="2TpxZ7JUBn3uw46aR7qd6V")
    a = build_demo_response(entity, limit=4)
    b = build_demo_response(entity, limit=4)
    assert [r.name for r in a.recommendations] == [r.name for r in b.recommendations]
    assert [r.similarity for r in a.recommendations] == [r.similarity for r in b.recommendations]
