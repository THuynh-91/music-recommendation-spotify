from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pytest
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles

pytest.importorskip("aiosqlite")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.db import models
from app.db.base import Base
from app.services.recommendations import (
    _backfill_with_spotify,
    _ensure_track_vector,
    _load_track_with_artists,
)


@compiles(JSONB, "sqlite")
def _compile_jsonb_to_sqlite(element, compiler, **kw):  # pragma: no cover - SQLite shim
    return "JSON"


class _StubIndexService:
    def __init__(self) -> None:
        self.vectors: Dict[str, np.ndarray] = {}

    async def ensure_loaded(self, session) -> None:  # pragma: no cover - compatibility shim
        return None

    async def add_vectors(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        for track_id, vector in items:
            self.vectors[track_id] = vector

    async def search(self, vector: np.ndarray, *, top_k: int) -> List[tuple[str, float]]:  # pragma: no cover - unused here
        return []


@dataclass(slots=True)
class _StubSpotifyClient:
    tracks: Dict[str, Dict[str, Any]]
    features: Dict[str, Dict[str, Any]]
    analysis: Dict[str, Dict[str, Any]]
    artists: Dict[str, Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    related: Dict[str, List[Dict[str, Any]]] | None = None
    last_request: Dict[str, Any] | None = None

    async def get_track(self, track_id: str) -> Dict[str, Any]:
        return self.tracks[track_id]

    async def get_audio_features(self, track_id: str) -> Dict[str, Any]:
        return self.features[track_id]

    async def get_audio_analysis(self, track_id: str) -> Dict[str, Any]:
        return self.analysis[track_id]

    async def get_artists(self, artist_ids: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.artists[artist_id] for artist_id in artist_ids if artist_id in self.artists]

    async def get_related_artists(self, artist_id: str) -> Dict[str, Any]:
        related = []
        if self.related and artist_id in self.related:
            related = list(self.related[artist_id])
        return {"artists": related}

    async def get_recommendations(
        self,
        *,
        seed_tracks: Sequence[str] | None = None,
        seed_artists: Sequence[str] | None = None,
        seed_genres: Sequence[str] | None = None,
        limit: int = 20,
        tunable_params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:  # pragma: no cover - parameters unused in stub
        self.last_request = {
            "seed_tracks": list(seed_tracks or []),
            "seed_artists": list(seed_artists or []),
            "seed_genres": list(seed_genres or []),
            "limit": limit,
            "tunable_params": dict(tunable_params or {}),
        }
        return {"tracks": list(self.recommendations)}


async def _run_backfill_flow(tmp_path: Path) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with maker() as session:
        seed_track_id = "seed-track"
        recommendation_id = "rec-track"

        seed_artist_id = "seed-artist"
        rec_artist_id = "rec-artist"

        stub_client = _StubSpotifyClient(
            tracks={
                seed_track_id: {
                    "id": seed_track_id,
                    "name": "Seed Song",
                    "duration_ms": 180000,
                    "explicit": False,
                    "popularity": 40,
                    "preview_url": None,
                    "uri": "spotify:track:seed",
                    "external_urls": {"spotify": "https://open.spotify.com/track/seed"},
                    "album": {
                        "id": "seed-album",
                        "name": "Seed Album",
                        "images": [{"url": "https://example.com/seed.jpg"}],
                    },
                    "artists": [
                        {
                            "id": seed_artist_id,
                            "name": "Seed Artist",
                        }
                    ],
                }
            },
            features={
                seed_track_id: {
                    "id": seed_track_id,
                    "danceability": 0.62,
                    "energy": 0.41,
                    "speechiness": 0.05,
                    "acousticness": 0.6,
                    "instrumentalness": 0.0,
                    "liveness": 0.12,
                    "valence": 0.34,
                    "tempo": 96.0,
                    "mode": 1,
                    "key": 5,
                    "loudness": -8.0,
                    "duration_ms": 180000,
                },
                recommendation_id: {
                    "id": recommendation_id,
                    "danceability": 0.64,
                    "energy": 0.39,
                    "speechiness": 0.04,
                    "acousticness": 0.58,
                    "instrumentalness": 0.0,
                    "liveness": 0.15,
                    "valence": 0.36,
                    "tempo": 95.0,
                    "mode": 1,
                    "key": 5,
                    "loudness": -9.0,
                    "duration_ms": 182000,
                },
            },
            analysis={
                seed_track_id: {
                    "track": {
                        "tempo_confidence": 0.8,
                        "time_signature": 4,
                        "time_signature_confidence": 0.7,
                        "key_confidence": 0.5,
                    },
                    "sections": [],
                },
                recommendation_id: {
                    "track": {
                        "tempo_confidence": 0.82,
                        "time_signature": 4,
                        "time_signature_confidence": 0.68,
                        "key_confidence": 0.45,
                    },
                    "sections": [],
                },
            },
            artists={
                seed_artist_id: {
                    "id": seed_artist_id,
                    "name": "Seed Artist",
                    "genres": ["viet r&b"],
                },
                rec_artist_id: {
                    "id": rec_artist_id,
                    "name": "Rec Artist",
                    "genres": ["viet r&b", "asian r&b"],
                },
            },
            recommendations=[
                {
                    "id": recommendation_id,
                    "name": "Recommended Song",
                    "duration_ms": 182000,
                    "explicit": False,
                    "popularity": 45,
                    "preview_url": None,
                    "uri": "spotify:track:rec",
                    "external_urls": {"spotify": "https://open.spotify.com/track/rec"},
                    "album": {
                        "id": "rec-album",
                        "name": "Rec Album",
                        "images": [{"url": "https://example.com/rec.jpg"}],
                    },
                    "artists": [
                        {
                            "id": rec_artist_id,
                            "name": "Rec Artist",
                        }
                    ],
                }
            ],
            related={
                seed_artist_id: [
                    {
                        "id": rec_artist_id,
                        "name": "Rec Artist",
                    }
                ]
            },
        )

        settings = Settings(
            faiss_index_path=tmp_path / "index.faiss",
            faiss_meta_path=tmp_path / "index.json",
            dsp_preview_timeout=0,
        )
        index_service = _StubIndexService()

        seed_track, seed_vector, seed_features, _ = await _ensure_track_vector(
            session,
            stub_client,
            index_service,
            settings,
            track_payload=await stub_client.get_track(seed_track_id),
        )

        seed_track_loaded = await _load_track_with_artists(session, seed_track.id)
        assert seed_track_loaded is not None

        recommendations = await _backfill_with_spotify(
            session,
            stub_client,
            index_service,
            settings,
            seed_track=seed_track_loaded,
            seed_vector=seed_vector,
            seed_features=seed_features,
            seed_genres={"viet r&b"},
            seed_genre_priority=["viet r&b"],
            existing=[],
            limit=3,
            extra_seed_artists=[rec_artist_id],
        )

        assert any(item.track_id == recommendation_id for item in recommendations)

        ingested_track = await session.get(models.Track, recommendation_id)
        assert ingested_track is not None
        assert {artist.id for artist in ingested_track.artists} == {rec_artist_id}

    await engine.dispose()

    return stub_client


def test_spotify_backfill_ingests_unseen_tracks(tmp_path: Path) -> None:
    stub_client = asyncio.run(_run_backfill_flow(tmp_path))
    assert stub_client.last_request is not None
    params = stub_client.last_request["tunable_params"]
    seed_artists = stub_client.last_request["seed_artists"]
    assert seed_artists[0] == "seed-artist"
    assert "rec-artist" in seed_artists
    assert pytest.approx(params["target_tempo"], rel=0.0, abs=0.01) == 96.0
    assert params["min_tempo"] < params["target_tempo"] < params["max_tempo"]
    assert pytest.approx(params["target_danceability"], rel=0.0, abs=0.001) == 0.62
    assert params["min_danceability"] < params["target_danceability"] < params["max_danceability"]
