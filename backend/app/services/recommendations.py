from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import logging
import numpy as np
from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
from ..spotify.client import SpotifyClient, SpotifyClientError
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


async def _load_track_with_artists(session: AsyncSession, track_id: str) -> Optional[models.Track]:
    result = await session.execute(
        select(models.Track).options(selectinload(models.Track.artists), selectinload(models.Track.album)).where(models.Track.id == track_id)
    )
    return result.scalars().first()



logger = logging.getLogger("recommendations")

DEFAULT_AUDIO_FEATURE_BASE = {
    "danceability": 0.55,
    "energy": 0.6,
    "speechiness": 0.05,
    "acousticness": 0.25,
    "instrumentalness": 0.0,
    "liveness": 0.18,
    "valence": 0.5,
    "tempo": 120.0,
    "key": -1.0,
    "mode": 1.0,
    "loudness": -11.0,
}

DEFAULT_AUDIO_ANALYSIS_TRACK = {
    "tempo_confidence": 0.0,
    "time_signature": 4.0,
    "time_signature_confidence": 0.0,
    "key_confidence": 0.0,
}


def _fallback_audio_features(track_payload: Dict[str, Any] | None) -> Dict[str, float]:
    fallback = DEFAULT_AUDIO_FEATURE_BASE.copy()
    duration = 0.0
    if isinstance(track_payload, dict):
        try:
            duration = float(track_payload.get("duration_ms") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    fallback["duration_ms"] = duration
    return fallback


def _normalize_audio_features(track_id: str, raw_features: Dict[str, Any] | None, track_payload: Dict[str, Any] | None) -> Tuple[Dict[str, Any], bool]:
    fallback = _fallback_audio_features(track_payload)
    sanitized: Dict[str, Any] = dict(raw_features or {})
    sanitized.setdefault("id", track_id)
    used_defaults = False
    for key, default in fallback.items():
        value = sanitized.get(key)
        if value is None:
            sanitized[key] = default
            used_defaults = True
            continue
        if isinstance(value, (int, float)):
            continue
        try:
            sanitized[key] = float(value)
        except (TypeError, ValueError):
            sanitized[key] = default
            used_defaults = True
    return sanitized, used_defaults


def _normalize_audio_analysis(raw_analysis: Dict[str, Any] | None) -> Tuple[Dict[str, Any], bool]:
    sanitized: Dict[str, Any] = dict(raw_analysis or {})
    track_meta = dict(sanitized.get("track") or {})
    used_defaults = False
    for key, default in DEFAULT_AUDIO_ANALYSIS_TRACK.items():
        value = track_meta.get(key)
        if value is None:
            track_meta[key] = default
            used_defaults = True
            continue
        if isinstance(value, (int, float)):
            try:
                track_meta[key] = float(value)
            except (TypeError, ValueError):
                track_meta[key] = default
                used_defaults = True
            continue
        try:
            track_meta[key] = float(value)
        except (TypeError, ValueError):
            track_meta[key] = default
            used_defaults = True
    sanitized["track"] = track_meta
    if not isinstance(sanitized.get("sections"), list):
        sanitized["sections"] = []
        used_defaults = True
    return sanitized, used_defaults

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


def _compute_similarity_score(raw_score: float, genre_overlap: int, seed_genre_count: int) -> float:
    base = max(min((raw_score + 1.0) / 2.0, 1.0), 0.0)
    if seed_genre_count > 0:
        if genre_overlap > 0:
            boost = 0.06 + max(genre_overlap - 1, 0) * 0.04
            base += min(boost, 0.22)
        else:
            base -= 0.18
    elif genre_overlap > 0:
        base += min(genre_overlap * 0.03, 0.1)
    return max(min(base, 1.0), 0.0)


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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


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

    features_source = audio_features
    if features_source is None:
        try:
            features_source = await spotify_client.get_audio_features(track_id)
        except SpotifyClientError as exc:
            logger.warning("Audio features unavailable for %s: %s", track_id, exc)
            features_source = {}
    features_payload, features_defaulted = _normalize_audio_features(track_id, features_source, track_payload)

    analysis_source = audio_analysis
    if analysis_source is None:
        try:
            analysis_source = await spotify_client.get_audio_analysis(track_id)
        except SpotifyClientError as exc:
            logger.warning("Audio analysis unavailable for %s: %s", track_id, exc)
            analysis_source = {}
    analysis_payload, analysis_defaulted = _normalize_audio_analysis(analysis_source)
    if features_defaulted:
        logger.debug("Using fallback audio features for %s", track_id)
    if analysis_defaulted:
        logger.debug("Using fallback audio analysis for %s", track_id)

    dsp_features: Dict[str, Any] | None = await compute_dsp_features(track.preview_url, timeout=settings.dsp_preview_timeout)
    vector = build_feature_vector(features_payload, analysis_payload, dsp_features)
    if not np.any(vector):
        vector = np.zeros_like(vector)
        if vector.size:
            vector[0] = 1.0
    await _upsert_track_feature(session, track, vector, features_payload, analysis_payload, dsp_features)
    if update_index:
        try:
            await index_service.add_vectors([(track.id, vector)])
        except Exception as exc:
            logger.warning("Failed to update FAISS index for %s: %s", track_id, exc)
    return track, vector, features_payload, True


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

    seed_genre_count = len(seed_genres)
    ranked: List[Tuple[float, models.Track, models.TrackFeature, SharedGenres, int]] = []
    for track_id, raw_score in matches:
        if track_id in exclude:
            continue
        data = track_map.get(track_id)
        if not data:
            continue
        track_obj, feature_obj = data
        candidate_genres = _collect_genres(track_obj.artists)
        genre_overlap = len(seed_genres & candidate_genres)
        similarity = _compute_similarity_score(raw_score, genre_overlap, seed_genre_count)
        ranked.append((similarity, track_obj, feature_obj, candidate_genres, genre_overlap))

    ranked.sort(key=lambda item: (item[4] > 0, item[0]), reverse=True)

    final: List[RecommendationItem] = []
    seen_artists: Set[str] = set()
    for similarity, track_obj, feature_obj, candidate_genres, _ in ranked:
        if len(final) >= top_k:
            break
        artist_ids = {artist.id for artist in track_obj.artists}
        penalty = 0.08 if seen_artists & artist_ids else 0.0
        adjusted = max(similarity - penalty, 0.0)
        explanation = _build_explanation(seed_features, feature_obj.audio_features, seed_genres & candidate_genres)
        final.append(_track_to_recommendation(track_obj, adjusted, explanation))
        seen_artists.update(artist_ids)
    return final


async def _collect_seen_artist_ids(session: AsyncSession, track_ids: Set[str]) -> Set[str]:
    if not track_ids:
        return set()
    result = await session.execute(
        select(models.Track)
        .options(selectinload(models.Track.artists))
        .where(models.Track.id.in_(track_ids))
    )
    seen: Set[str] = set()
    for track in result.scalars():
        seen.update(artist.id for artist in track.artists if artist.id)
    return seen


async def _backfill_with_spotify(
    session: AsyncSession,
    spotify_client: SpotifyClient,
    index_service: FaissService,
    settings: Settings,
    *,
    seed_track: models.Track,
    seed_vector: np.ndarray,
    seed_features: Dict[str, Any],
    seed_genres: SharedGenres,
    existing: List[RecommendationItem],
    limit: int,
) -> List[RecommendationItem]:
    needed = limit - len(existing)
    if needed <= 0:
        return existing

    existing_ids = {item.track_id for item in existing}
    exclude_ids = existing_ids | {seed_track.id}
    seen_artists = await _collect_seen_artist_ids(session, existing_ids)

    artist_ids = [artist.id for artist in seed_track.artists if artist.id]
    seed_genre_list = sorted(seed_genres)
    try:
        rec_payload = await spotify_client.get_recommendations(
            seed_tracks=[seed_track.id],
            seed_artists=artist_ids[:2],
            seed_genres=seed_genre_list[:2],
            limit=min(max(needed * 2, needed + 2), settings.recommendation_max_limit * 2),
        )
    except SpotifyClientError as exc:
        logger.warning("Spotify fallback recommendations failed for %s: %s", seed_track.id, exc)
        return existing

    tracks_payload = rec_payload.get("tracks") if isinstance(rec_payload, dict) else None
    if not tracks_payload:
        return existing

    seed_genre_count = len(seed_genres)
    augmented: List[Tuple[int, float, RecommendationItem, Set[str]]] = []
    for track_payload in tracks_payload:
        track_id = track_payload.get("id") if isinstance(track_payload, dict) else None
        if not track_id or track_id in exclude_ids:
            continue
        try:
            candidate_track, candidate_vector, candidate_features, _ = await _ensure_track_vector(
                session,
                spotify_client,
                index_service,
                settings,
                track_payload=track_payload,
            )
        except SpotifyClientError as exc:
            logger.debug("Failed to ingest recommended track %s: %s", track_id, exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unexpected error ingesting recommended track %s: %s", track_id, exc)
            continue

        track_with_relations = await _load_track_with_artists(session, candidate_track.id)
        if track_with_relations is not None:
            candidate_track = track_with_relations

        candidate_genres = _collect_genres(candidate_track.artists)
        raw_score = _cosine_similarity(seed_vector, candidate_vector)
        genre_overlap = len(seed_genres & candidate_genres)
        similarity = _compute_similarity_score(raw_score, genre_overlap, seed_genre_count)
        shared = seed_genres & candidate_genres
        explanation = _build_explanation(seed_features, candidate_features, shared)

        artist_ids_set = {artist.id for artist in candidate_track.artists if artist.id}
        penalty = 0.08 if seen_artists & artist_ids_set else 0.0
        adjusted = max(similarity - penalty, 0.0)

        if seed_genres and genre_overlap == 0 and adjusted < 0.55:
            # Skip obviously unrelated tracks when we know the seed genres
            continue

        item = _track_to_recommendation(candidate_track, adjusted, explanation)
        augmented.append((genre_overlap, adjusted, item, artist_ids_set))

    if not augmented:
        return existing

    augmented.sort(key=lambda entry: (entry[0] > 0, entry[0], entry[1]), reverse=True)

    final = list(existing)
    for genre_overlap, adjusted, item, artist_ids_set in augmented:
        if len(final) >= limit:
            break
        if item.track_id in exclude_ids:
            continue
        final.append(item)
        exclude_ids.add(item.track_id)
        seen_artists.update(artist_ids_set)

    return final


async def _process_playlist(
    playlist_id: str,
    *,
    session: AsyncSession,
    redis: Redis,
    spotify_client: SpotifyClient,
    index_service: FaissService,
    settings: Settings,
    top_k: int,
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
        if track_ids:
            try:
                audio_features_list = await spotify_client.get_audio_features_bulk(track_ids)
            except SpotifyClientError as exc:
                logger.warning("Bulk audio feature fetch failed for playlist %s: %s", playlist_id, exc)
                audio_features_list = [{} for _ in track_ids]
        else:
            audio_features_list = []
        audio_features_map: Dict[str, Dict[str, Any]] = {}
        for idx, track_id in enumerate(track_ids):
            raw_features = audio_features_list[idx] if idx < len(audio_features_list) else {}
            track_payload = items[idx][1]
            normalized, _ = _normalize_audio_features(track_id, raw_features, track_payload)
            audio_features_map[track_id] = normalized

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
            search_k = max(top_k * 4, top_k)
            matches = await index_service.search(centroid, top_k=search_k)
            exclude = set(track_ids)
            seed_genres = set(genre for genre, _ in seed_genres_counter.most_common(20))
            seed_features = _average_audio_features(feature_payloads)
            recommendations = await _hydrate_recommendations(
                session,
                matches,
                exclude=exclude,
                seed_features=seed_features,
                seed_genres=seed_genres,
                top_k=top_k,
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
    limit: int,
) -> RecommendResponse:
    await index_service.ensure_loaded(session)

    if entity.kind == "track":
        track_payload = await spotify_client.get_track(entity.id)
        track, vector, features, _ = await _ensure_track_vector(
            session,
            spotify_client,
            index_service,
            settings,
            track_payload=track_payload,
        )

        track_with_relations = await _load_track_with_artists(session, track.id)
        if track_with_relations is not None:
            track = track_with_relations

        search_k = max(limit * 3, limit)
        matches = await index_service.search(vector, top_k=search_k)
        seed_genres = _collect_genres(track.artists)
        recommendations = await _hydrate_recommendations(
            session,
            matches,
            exclude={track.id},
            seed_features=features,
            seed_genres=seed_genres,
            top_k=limit,
        )
        if len(recommendations) < limit:
            recommendations = await _backfill_with_spotify(
                session,
                spotify_client,
                index_service,
                settings,
                seed_track=track,
                seed_vector=vector,
                seed_features=features,
                seed_genres=seed_genres,
                existing=recommendations,
                limit=limit,
            )
        response = RecommendResponse(
            type="track",
            seed_track=_track_to_seed(track),
            recommendations=recommendations,
        )
        await session.commit()
        return response

    if entity.kind == "playlist":
        return await _process_playlist(
            entity.id,
            session=session,
            redis=redis,
            spotify_client=spotify_client,
            index_service=index_service,
            settings=settings,
            top_k=limit,
        )

    raise ValueError(f"Unsupported entity type {entity.kind}")


