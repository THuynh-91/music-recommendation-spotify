from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings
from ..db import models
from ..schemas.recommend import (
    PlaylistIngestSummary,
    RecommendResponse,
    RecommendationItem,
    SeedPlaylist,
    SeedTrack,
)
from ..services.features import build_feature_vector, compute_dsp_features
from ..services.index import FaissService
from ..spotify.client import SpotifyClient
from ..spotify.parsing import SpotifyEntity

AUDIO_FEATURE_KEYS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "mode",
]

SharedGenres = Set[str]


def _first_image(images: Iterable[Dict[str, Any]] | None) -> Optional[str]:
    if not images:
        return None
    return next((img.get("url") for img in images if img and img.get("url")), None)


async def _upsert_album(session: AsyncSession, payload: Dict[str, Any] | None) -> Optional[models.Album]:
    if not payload:
        return None
    album_id = payload.get("id")
    if not album_id:
        return None
    album = await session.get(models.Album, album_id)
    if album is None:
        album = models.Album(id=album_id, name=payload.get("name") or "Unknown Album")
        session.add(album)
    album.name = payload.get("name") or album.name
    album.release_date = payload.get("release_date")
    album.release_date_precision = payload.get("release_date_precision")
    album.image_url = _first_image(payload.get("images")) or album.image_url
    await session.flush()
    return album


async def _upsert_artists(
    session: AsyncSession,
    payload: Sequence[Dict[str, Any]] | None,
    spotify_client: SpotifyClient,
) -> List[models.Artist]:
    if not payload:
        return []
    payload_by_id = {artist.get("id"): artist for artist in payload if artist and artist.get("id")}
    artist_ids = list(payload_by_id.keys())
    existing: Dict[str, Optional[models.Artist]] = {}
    for artist_id in artist_ids:
        existing[artist_id] = await session.get(models.Artist, artist_id)

    missing_ids = [artist_id for artist_id, artist in existing.items() if artist is None]
    remote_details: Dict[str, Dict[str, Any]] = {}
    if missing_ids:
        remote = await spotify_client.get_artists(missing_ids)
        remote_details = {artist.get("id"): artist for artist in remote if artist.get("id")}

    result: List[models.Artist] = []
    for artist_id in artist_ids:
        artist_obj = existing.get(artist_id)
        base_payload = payload_by_id[artist_id]
        if artist_obj is None:
            artist_obj = models.Artist(id=artist_id, name=base_payload.get("name") or "Unknown Artist")
        artist_obj.name = base_payload.get("name") or artist_obj.name
        remote_payload = remote_details.get(artist_id)
        genres = remote_payload.get("genres") if remote_payload else base_payload.get("genres")
        if genres is not None:
            artist_obj.genres = genres
        popularity = remote_payload.get("popularity") if remote_payload else base_payload.get("popularity")
        if popularity is not None:
            artist_obj.popularity = popularity
        session.add(artist_obj)
        result.append(artist_obj)
    await session.flush()
    return result


async def _upsert_track(
    session: AsyncSession,
    payload: Dict[str, Any],
    spotify_client: SpotifyClient,
) -> models.Track:
    track_id = payload.get("id")
    if not track_id:
        raise ValueError("track payload missing id")
    track = await session.get(models.Track, track_id)
    if track is None:
        track = models.Track(id=track_id, name=payload.get("name") or "Untitled")

    album_payload = payload.get("album")
    album = await _upsert_album(session, album_payload)
    artists = await _upsert_artists(session, payload.get("artists"), spotify_client)

    track.name = payload.get("name") or track.name
    track.duration_ms = payload.get("duration_ms")
    track.explicit = bool(payload.get("explicit"))
    popularity = payload.get("popularity")
    if popularity is not None:
        track.popularity = popularity
    track.preview_url = payload.get("preview_url")
    track.external_url = (payload.get("external_urls") or {}).get("spotify")
    track.uri = payload.get("uri")
    track.album = album
    if album_payload:
        track.image_url = _first_image(album_payload.get("images")) or track.image_url
    track.artists = artists
    session.add(track)
    await session.flush()
    return track


async def _upsert_track_feature(
    session: AsyncSession,
    track: models.Track,
    vector: np.ndarray,
    audio_features: Dict[str, Any],
    audio_analysis: Dict[str, Any] | None,
    dsp_features: Dict[str, Any] | None,
) -> models.TrackFeature:
    feature = await session.get(models.TrackFeature, track.id)
    if feature is None:
        feature = models.TrackFeature(track_id=track.id, vector=b"", vector_dim=vector.shape[0], audio_features={})
    feature.vector = vector.astype(np.float32).tobytes()
    feature.vector_dim = vector.shape[0]
    feature.audio_features = audio_features
    feature.audio_analysis = audio_analysis
    feature.dsp_features = dsp_features
    feature.track = track
    session.add(feature)
    await session.flush()
    return feature


def _vector_from_feature(feature: models.TrackFeature) -> np.ndarray:
    return np.frombuffer(feature.vector, dtype=np.float32)


def _collect_genres(artists: Sequence[models.Artist]) -> SharedGenres:
    genres: SharedGenres = set()
    for artist in artists:
        if artist.genres:
            genres.update({g.lower() for g in artist.genres})
    return genres


def _compute_similarity_score(raw_score: float, genre_overlap: int) -> float:
    base = max(min((raw_score + 1.0) / 2.0, 1.0), 0.0)
    boost = min(genre_overlap * 0.04, 0.12)
    return max(min(base + boost, 1.0), 0.0)


def _build_explanation(
    seed_features: Dict[str, Any],
    candidate_features: Dict[str, Any],
    shared_genres: SharedGenres,
) -> str:
    highlights: List[str] = []
    tempo_seed = seed_features.get("tempo")
    tempo_candidate = candidate_features.get("tempo")
    if tempo_seed is not None and tempo_candidate is not None and abs(tempo_seed - tempo_candidate) <= 8:
        highlights.append(f"matching tempo around {round(float(tempo_candidate))} BPM")

    for field, label in (("danceability", "danceability"), ("energy", "energy"), ("valence", "mood")):
        seed_val = seed_features.get(field)
        cand_val = candidate_features.get(field)
        if seed_val is None or cand_val is None:
            continue
        diff = abs(float(seed_val) - float(cand_val))
        if diff <= 0.12:
            if seed_val >= 0.7 and cand_val >= 0.7:
                descriptor = "high"
            elif seed_val <= 0.3 and cand_val <= 0.3:
                descriptor = "low"
            else:
                descriptor = "similar"
            highlights.append(f"{descriptor} {label}")
    if shared_genres:
        top_genres = ", ".join(sorted(shared_genres)[:2])
        highlights.append(f"shares {top_genres} vibes")

    if not highlights:
        return "Overall sonic profile aligns closely"
    if len(highlights) == 1:
        return highlights[0].capitalize()
    return highlights[0].capitalize() + "; " + "; ".join(highlights[1:])


def _average_audio_features(features_list: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not features_list:
        return {}
    totals: Dict[str, float] = {key: 0.0 for key in AUDIO_FEATURE_KEYS}
    counts: Dict[str, int] = {key: 0 for key in AUDIO_FEATURE_KEYS}
    for features in features_list:
        for key in AUDIO_FEATURE_KEYS:
            value = features.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                totals[key] += float(value)
                counts[key] += 1
    averages = {
        key: totals[key] / counts[key]
        for key in AUDIO_FEATURE_KEYS
        if counts[key]
    }
    return averages


async def _ensure_track_vector(
    session: AsyncSession,
    spotify_client: SpotifyClient,
    index_service: FaissService,
    settings: Settings,
    *,
    track_payload: Dict[str, Any] | None = None,
    audio_features: Dict[str, Any] | None = None,
    audio_analysis: Dict[str, Any] | None = None,
    update_index: bool = True,
) -> Tuple[models.Track, np.ndarray, Dict[str, Any], bool]:
    track_id = (track_payload or {}).get("id") or (audio_features or {}).get("id")
    if not track_id:
        raise ValueError("insufficient data to identify track")

    track = await session.get(models.Track, track_id)
    feature = await session.get(models.TrackFeature, track_id)
    if track and feature:
        vector = _vector_from_feature(feature)
        return track, vector, feature.audio_features, False

    if track_payload is None:
        track_payload = await spotify_client.get_track(track_id)
    track = await _upsert_track(session, track_payload, spotify_client)

    if audio_features is None:
        audio_features = await spotify_client.get_audio_features(track_id)
    if audio_analysis is None:
        audio_analysis = await spotify_client.get_audio_analysis(track_id)

    dsp_features: Dict[str, Any] | None = await compute_dsp_features(track.preview_url, timeout=settings.dsp_preview_timeout)
    vector = build_feature_vector(audio_features, audio_analysis, dsp_features)
    await _upsert_track_feature(session, track, vector, audio_features, audio_analysis, dsp_features)
    if update_index:
        await index_service.add_vectors([(track.id, vector)])
    return track, vector, audio_features, True


def _track_to_seed(track: models.Track) -> SeedTrack:
    return SeedTrack(
        id=track.id,
        name=track.name,
        artists=[artist.name for artist in track.artists],
        external_url=track.external_url,
        image_url=track.image_url,
    )


def _track_to_recommendation(
    track: models.Track,
    similarity: float,
    explanation: str,
) -> RecommendationItem:
    return RecommendationItem(
        track_id=track.id,
        name=track.name,
        artists=[artist.name for artist in track.artists],
        preview_url=track.preview_url,
        external_url=track.external_url,
        image_url=track.image_url,
        similarity=similarity,
        explanation=explanation,
    )


async def _hydrate_recommendations(
    session: AsyncSession,
    matches: Sequence[Tuple[str, float]],
    *,
    exclude: Set[str],
    seed_features: Dict[str, Any],
    seed_genres: SharedGenres,
    top_k: int,
) -> List[RecommendationItem]:
    if not matches:
        return []
    track_ids = [track_id for track_id, _ in matches if track_id not in exclude]
    if not track_ids:
        return []

    result = await session.execute(
        select(models.Track, models.TrackFeature)
        .join(models.TrackFeature, models.Track.id == models.TrackFeature.track_id)
        .where(models.Track.id.in_(track_ids))
    )
    track_map: Dict[str, Tuple[models.Track, models.TrackFeature]] = {
        row[0].id: (row[0], row[1]) for row in result.all()
    }

    ranked: List[Tuple[float, models.Track, models.TrackFeature, SharedGenres]] = []
    for track_id, raw_score in matches:
        if track_id in exclude:
            continue
        data = track_map.get(track_id)
        if not data:
            continue
        track_obj, feature_obj = data
        candidate_genres = _collect_genres(track_obj.artists)
        genre_overlap = len(seed_genres & candidate_genres)
        similarity = _compute_similarity_score(raw_score, genre_overlap)
        ranked.append((similarity, track_obj, feature_obj, candidate_genres))

    ranked.sort(key=lambda item: item[0], reverse=True)

    final: List[RecommendationItem] = []
    seen_artists: Set[str] = set()
    for similarity, track_obj, feature_obj, candidate_genres in ranked:
        if len(final) >= top_k:
            break
        artist_ids = {artist.id for artist in track_obj.artists}
        penalty = 0.08 if seen_artists & artist_ids else 0.0
        adjusted = max(similarity - penalty, 0.0)
        explanation = _build_explanation(seed_features, feature_obj.audio_features, seed_genres & candidate_genres)
        final.append(_track_to_recommendation(track_obj, adjusted, explanation))
        seen_artists.update(artist_ids)
    return final


async def _process_playlist(
    playlist_id: str,
    *,
    session: AsyncSession,
    redis: Redis,
    spotify_client: SpotifyClient,
    index_service: FaissService,
    settings: Settings,
) -> RecommendResponse:
    lock_key = f"playlist:{playlist_id}:ingest"
    lock_acquired = await redis.set(lock_key, "1", nx=True, ex=600)
    try:
        playlist_meta = await spotify_client.get_playlist(playlist_id)
        playlist = await session.get(models.Playlist, playlist_id)
        if playlist is None:
            playlist = models.Playlist(id=playlist_id, name=playlist_meta.get("name") or "Playlist")
        playlist.name = playlist_meta.get("name") or playlist.name
        playlist.description = playlist_meta.get("description")
        owner = playlist_meta.get("owner") or {}
        playlist.owner_id = owner.get("id")
        playlist.owner_display_name = owner.get("display_name")
        playlist.snapshot_id = playlist_meta.get("snapshot_id")
        playlist.image_url = _first_image(playlist_meta.get("images")) or playlist.image_url
        playlist.last_ingested_at = datetime.utcnow()
        session.add(playlist)

        items: List[Tuple[int, Dict[str, Any]]] = []
        position = 0
        async for entry in spotify_client.iter_playlist_tracks(playlist_id, batch_size=settings.playlist_ingest_batch_size):
            track_payload = entry.get("track") if entry else None
            if not track_payload or not track_payload.get("id"):
                continue
            items.append((position, track_payload))
            position += 1

        track_ids = [payload.get("id") for _, payload in items]
        audio_features_list = await spotify_client.get_audio_features_bulk(track_ids)
        audio_features_map = {item.get("id"): item for item in audio_features_list if item and item.get("id")}

        vectors: List[np.ndarray] = []
        feature_payloads: List[Dict[str, Any]] = []
        seed_genres_counter: Counter[str] = Counter()
        track_objects: Dict[str, models.Track] = {}
        new_vectors: List[Tuple[str, np.ndarray]] = []

        for _, track_payload in items:
            track_id = track_payload.get("id")
            features_payload = audio_features_map.get(track_id)
            track_obj, vector, features, updated = await _ensure_track_vector(
                session,
                spotify_client,
                index_service,
                settings,
                track_payload=track_payload,
                audio_features=features_payload,
                update_index=False,
            )
            vectors.append(vector)
            feature_payloads.append(features)
            seed_genres_counter.update(_collect_genres(track_obj.artists))
            track_objects[track_obj.id] = track_obj
            if updated:
                new_vectors.append((track_obj.id, vector))

        await session.execute(delete(models.PlaylistTrack).where(models.PlaylistTrack.playlist_id == playlist_id))
        for position, track_payload in items:
            track_obj = track_objects.get(track_payload.get("id"))
            if not track_obj:
                continue
            playlist_track = models.PlaylistTrack(
                playlist_id=playlist_id,
                track_id=track_obj.id,
                position=position,
            )
            session.add(playlist_track)

        await session.commit()
        if new_vectors:
            await index_service.add_vectors(new_vectors)

        centroid = None
        if vectors:
            centroid = np.mean(np.stack(vectors), axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = (centroid / norm).astype(np.float32)

        recommendations: List[RecommendationItem] = []
        if centroid is not None:
            await index_service.ensure_loaded(session)
            matches = await index_service.search(centroid, top_k=settings.recommendation_top_k * 4)
            exclude = set(track_ids)
            seed_genres = set(genre for genre, _ in seed_genres_counter.most_common(20))
            seed_features = _average_audio_features(feature_payloads)
            recommendations = await _hydrate_recommendations(
                session,
                matches,
                exclude=exclude,
                seed_features=seed_features,
                seed_genres=seed_genres,
                top_k=settings.recommendation_top_k,
            )

        summary = PlaylistIngestSummary(
            id=playlist.id,
            name=playlist.name,
            track_count=len(items),
            ingested_tracks=len(vectors),
            snapshot_id=playlist.snapshot_id,
        )
        seed_playlist = SeedPlaylist(
            id=playlist.id,
            name=playlist.name,
            description=playlist.description,
            owner=playlist.owner_display_name or playlist.owner_id,
            image_url=playlist.image_url,
            track_count=len(items),
        )
        return RecommendResponse(
            type="playlist",
            seed_playlist=seed_playlist,
            playlist=summary,
            recommendations=recommendations,
        )
    finally:
        if lock_acquired:
            await redis.delete(lock_key)


async def recommend_for_entity(
    entity: SpotifyEntity,
    *,
    session: AsyncSession,
    redis: Redis,
    spotify_client: SpotifyClient,
    index_service: FaissService,
    settings: Settings,
) -> RecommendResponse:
    await index_service.ensure_loaded(session)

    if entity.kind == "track":
        track_payload = await spotify_client.get_track(entity.id)
        audio_features = await spotify_client.get_audio_features(entity.id)
        track, vector, features, _ = await _ensure_track_vector(
            session,
            spotify_client,
            index_service,
            settings,
            track_payload=track_payload,
            audio_features=audio_features,
        )
        await session.commit()

        matches = await index_service.search(vector, top_k=settings.recommendation_top_k * 3)
        seed_genres = _collect_genres(track.artists)
        recommendations = await _hydrate_recommendations(
            session,
            matches,
            exclude={track.id},
            seed_features=features,
            seed_genres=seed_genres,
            top_k=settings.recommendation_top_k,
        )
        return RecommendResponse(
            type="track",
            seed_track=_track_to_seed(track),
            recommendations=recommendations,
        )

    if entity.kind == "playlist":
        return await _process_playlist(
            entity.id,
            session=session,
            redis=redis,
            spotify_client=spotify_client,
            index_service=index_service,
            settings=settings,
        )

    raise ValueError(f"Unsupported entity type {entity.kind}")