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

RECOMMENDATION_FLOAT_CONSTRAINTS: Dict[str, Tuple[float, float, float]] = {
    "danceability": (0.0, 1.0, 0.12),
    "energy": (0.0, 1.0, 0.15),
    "valence": (0.0, 1.0, 0.15),
    "acousticness": (0.0, 1.0, 0.2),
    "instrumentalness": (0.0, 1.0, 0.2),
    "liveness": (0.0, 1.0, 0.18),
    "speechiness": (0.0, 1.0, 0.1),
}

TEMPO_RANGE = (30.0, 220.0, 8.0)
LOUDNESS_RANGE = (-60.0, 0.0, 5.0)
DURATION_RANGE = (30000.0, 600000.0, 20000.0)

FEATURE_ALIGNMENT_RULES: Sequence[Tuple[str, float, float]] = (
    ("danceability", 0.18, 1.0),
    ("energy", 0.22, 1.0),
    ("valence", 0.22, 0.95),
    ("acousticness", 0.28, 0.7),
    ("instrumentalness", 0.3, 0.55),
    ("speechiness", 0.18, 0.4),
)
TEMPO_ALIGNMENT_TOLERANCE = 18.0
LOUDNESS_ALIGNMENT_TOLERANCE = 6.0
DURATION_ALIGNMENT_TOLERANCE = 60000.0
MIN_ALIGNMENT_SCORE = 0.38

SharedGenres = Set[str]


GENDER_GENRE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "female": ("female", "women", "woman", "girl", "girls", "femme", "ladies", "diva"),
    "male": ("male", "men", "man", "boy", "boys", "lads", "king"),
}
GENDER_NAME_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "female": (" lady ", " queen ", " miss ", " mrs ", " ms ", " girl", "girls"),
    "male": (" mr ", " sir ", " king ", " prince ", " boy", "boys", " man", "men"),
}

def _infer_artist_gender(artists: Sequence[models.Artist]) -> Optional[str]:
    found: Set[str] = set()
    for artist in artists:
        genres = [genre.lower() for genre in (artist.genres or [])]
        for genre in genres:
            for gender, keywords in GENDER_GENRE_KEYWORDS.items():
                if any(keyword in genre for keyword in keywords):
                    found.add(gender)
        name = f" {(artist.name or '').lower()} "
        for gender, keywords in GENDER_NAME_KEYWORDS.items():
            if any(keyword in name for keyword in keywords):
                found.add(gender)
    if len(found) == 1:
        return next(iter(found))
    return None

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


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_spotify_tunable_params(
    seed_features: Dict[str, Any],
    *,
    seed_track: models.Track | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    for field, (lower, upper, window) in RECOMMENDATION_FLOAT_CONSTRAINTS.items():
        numeric = _coerce_float(seed_features.get(field))
        if numeric is None:
            continue
        clamped = max(lower, min(numeric, upper))
        params[f"target_{field}"] = round(clamped, 3)
        params[f"min_{field}"] = round(max(lower, clamped - window), 3)
        params[f"max_{field}"] = round(min(upper, clamped + window), 3)

    tempo = _coerce_float(seed_features.get("tempo"))
    if tempo is not None:
        tempo_min, tempo_max, tempo_window = TEMPO_RANGE
        tempo_clamped = max(tempo_min, min(tempo, tempo_max))
        params["target_tempo"] = round(tempo_clamped, 2)
        params["min_tempo"] = round(max(tempo_min, tempo_clamped - tempo_window), 2)
        params["max_tempo"] = round(min(tempo_max, tempo_clamped + tempo_window), 2)

    duration = _coerce_float(seed_features.get("duration_ms"))
    if duration is None and seed_track and seed_track.duration_ms is not None:
        duration = float(seed_track.duration_ms)
    if duration is not None:
        duration_min, duration_max, duration_window = DURATION_RANGE
        dynamic_window = max(duration_window, duration * 0.15)
        duration_clamped = max(duration_min, min(duration, duration_max))
        params["target_duration_ms"] = int(round(duration_clamped))
        params["min_duration_ms"] = int(max(duration_min, duration_clamped - dynamic_window))
        params["max_duration_ms"] = int(min(duration_max, duration_clamped + dynamic_window))

    loudness = _coerce_float(seed_features.get("loudness"))
    if loudness is not None:
        loud_min, loud_max, loud_window = LOUDNESS_RANGE
        loud_clamped = max(loud_min, min(loudness, loud_max))
        params["target_loudness"] = round(loud_clamped, 2)
        params["min_loudness"] = round(max(loud_min, loud_clamped - loud_window), 2)
        params["max_loudness"] = round(min(loud_max, loud_clamped + loud_window), 2)

    mode = seed_features.get("mode")
    if isinstance(mode, (int, float)):
        params["target_mode"] = 1 if int(round(mode)) else 0

    key = seed_features.get("key")
    if isinstance(key, (int, float)):
        key_int = int(round(key)) % 12
        params["target_key"] = key_int

    popularity = getattr(seed_track, "popularity", None)
    if popularity is not None:
        popularity_int = int(popularity)
        params["min_popularity"] = max(0, popularity_int - 15)
        params["max_popularity"] = min(100, popularity_int + 15)

    return params


def _compute_feature_alignment(
    seed_features: Dict[str, Any],
    candidate_features: Dict[str, Any],
) -> float:
    score = 0.0
    total_weight = 0.0

    for field, tolerance, weight in FEATURE_ALIGNMENT_RULES:
        seed_value = _coerce_float(seed_features.get(field))
        candidate_value = _coerce_float(candidate_features.get(field))
        if seed_value is None or candidate_value is None:
            continue
        diff = abs(seed_value - candidate_value)
        closeness = max(0.0, 1.0 - min(diff / tolerance, 1.0))
        score += closeness * weight
        total_weight += weight

    tempo_seed = _coerce_float(seed_features.get("tempo"))
    tempo_candidate = _coerce_float(candidate_features.get("tempo"))
    if tempo_seed is not None and tempo_candidate is not None:
        tempo_diff = abs(tempo_seed - tempo_candidate)
        closeness = max(0.0, 1.0 - min(tempo_diff / TEMPO_ALIGNMENT_TOLERANCE, 1.0))
        score += closeness * 1.1
        total_weight += 1.1

    loudness_seed = _coerce_float(seed_features.get("loudness"))
    loudness_candidate = _coerce_float(candidate_features.get("loudness"))
    if loudness_seed is not None and loudness_candidate is not None:
        loudness_diff = abs(loudness_seed - loudness_candidate)
        closeness = max(0.0, 1.0 - min(loudness_diff / LOUDNESS_ALIGNMENT_TOLERANCE, 1.0))
        score += closeness * 0.6
        total_weight += 0.6

    duration_seed = _coerce_float(seed_features.get("duration_ms"))
    duration_candidate = _coerce_float(candidate_features.get("duration_ms"))
    if duration_seed is not None and duration_candidate is not None:
        duration_diff = abs(duration_seed - duration_candidate)
        closeness = max(0.0, 1.0 - min(duration_diff / DURATION_ALIGNMENT_TOLERANCE, 1.0))
        score += closeness * 0.6
        total_weight += 0.6

    mode_seed = _coerce_float(seed_features.get("mode"))
    mode_candidate = _coerce_float(candidate_features.get("mode"))
    if mode_seed is not None and mode_candidate is not None:
        same_mode = 1.0 if int(round(mode_seed)) == int(round(mode_candidate)) else 0.0
        score += same_mode * 0.3
        total_weight += 0.3

    key_seed = _coerce_float(seed_features.get("key"))
    key_candidate = _coerce_float(candidate_features.get("key"))
    if key_seed is not None and key_candidate is not None:
        seed_int = int(round(key_seed)) % 12
        candidate_int = int(round(key_candidate)) % 12
        distance = abs(seed_int - candidate_int)
        distance = min(distance, 12 - distance)
        closeness = max(0.0, 1.0 - min(distance / 4.0, 1.0))
        score += closeness * 0.4
        total_weight += 0.4

    if total_weight == 0.0:
        return 0.5
    return max(min(score / total_weight, 1.0), 0.0)


def _compute_similarity_score(
    raw_score: float,
    genre_overlap: int,
    seed_genre_count: int,
    feature_alignment: float,
) -> float:
    base = max(min((raw_score + 1.0) / 2.0, 1.0), 0.0)
    combined = (base * 0.6) + (feature_alignment * 0.4)
    if seed_genre_count > 0:
        if genre_overlap > 0:
            boost = 0.07 + max(genre_overlap - 1, 0) * 0.035
            combined += min(boost, 0.2)
        else:
            combined -= 0.24
    elif genre_overlap > 0:
        combined += min(genre_overlap * 0.03, 0.09)
    return max(min(combined, 1.0), 0.0)


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
    ranked: List[Tuple[float, float, models.Track, models.TrackFeature, SharedGenres, int]] = []
    for track_id, raw_score in matches:
        if track_id in exclude:
            continue
        data = track_map.get(track_id)
        if not data:
            continue
        track_obj, feature_obj = data
        candidate_genres = _collect_genres(track_obj.artists)
        genre_overlap = len(seed_genres & candidate_genres)
        alignment = _compute_feature_alignment(seed_features, feature_obj.audio_features)
        if alignment < MIN_ALIGNMENT_SCORE and genre_overlap == 0:
            continue
        similarity = _compute_similarity_score(raw_score, genre_overlap, seed_genre_count, alignment)
        if similarity < MIN_ALIGNMENT_SCORE:
            continue
        ranked.append((similarity, alignment, track_obj, feature_obj, candidate_genres, genre_overlap))

    ranked.sort(key=lambda item: (item[5] > 0, item[0], item[1]), reverse=True)

    final: List[RecommendationItem] = []
    seen_artists: Set[str] = set()
    for similarity, _, track_obj, feature_obj, candidate_genres, _ in ranked:
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


async def _expand_artist_seeds(
    spotify_client: SpotifyClient,
    base_artist_ids: Sequence[str],
    *,
    limit: int = 3,
) -> List[str]:
    if limit <= 0:
        return []
    ordered_ids = [artist_id for artist_id in base_artist_ids if artist_id]
    if not ordered_ids:
        return []
    extras: List[str] = []
    seen: Set[str] = set(ordered_ids)
    for artist_id in ordered_ids[:5]:
        try:
            payload = await spotify_client.get_related_artists(artist_id)
        except SpotifyClientError:
            continue
        related = payload.get("artists") if isinstance(payload, dict) else None
        if not isinstance(related, list):
            continue
        for artist_payload in related:
            related_id = artist_payload.get("id") if isinstance(artist_payload, dict) else None
            if not related_id or related_id in seen:
                continue
            extras.append(related_id)
            seen.add(related_id)
            if len(extras) >= limit:
                return extras
    return extras



async def _get_spotify_recommendations(
    session: AsyncSession,
    spotify_client: SpotifyClient,
    *,
    seed_track: models.Track,
    seed_features: Dict[str, Any],
    seed_genres: SharedGenres,
    seed_genre_priority: Sequence[str] | None = None,
    limit: int,
    settings: Settings,
    index_service: FaissService,
    additional_exclude: Optional[Set[str]] = None,
    extra_seed_tracks: Sequence[str] | None = None,
    extra_seed_artists: Sequence[str] | None = None,
) -> List[RecommendationItem]:
    """Generate Spotify-powered recommendations and filter to close sonic matches."""
    if limit <= 0:
        return []

    exclude_ids: Set[str] = {seed_track.id}
    if additional_exclude:
        exclude_ids.update({track_id for track_id in additional_exclude if track_id})

    track_seeds: List[str] = []
    for track_id in [seed_track.id, *(extra_seed_tracks or [])]:
        if track_id and track_id not in track_seeds and track_id not in exclude_ids:
            track_seeds.append(track_id)
    track_seeds = track_seeds[:5]

    artist_seeds: List[str] = []
    for artist in seed_track.artists:
        if artist.id and artist.id not in artist_seeds:
            artist_seeds.append(artist.id)
    if extra_seed_artists:
        for artist_id in extra_seed_artists:
            if artist_id and artist_id not in artist_seeds:
                artist_seeds.append(artist_id)
    artist_seeds = artist_seeds[:5]

    seed_genre_list = [genre for genre in (seed_genre_priority or []) if genre]
    if not seed_genre_list:
        seed_genre_list = sorted(seed_genres)
    seed_genre_list = seed_genre_list[:5]

    tunable_params = _build_spotify_tunable_params(seed_features, seed_track=seed_track)
    request_limit = min(max(limit * 3, limit + 5), 100)

    logger.info(
        "Fetching Spotify recommendations for %s (artists=%s, genres=%s, limit=%s)",
        seed_track.id,
        artist_seeds,
        seed_genre_list,
        request_limit,
    )

    candidate_payloads: List[Dict[str, Any]] = []
    try:
        rec_payload = await spotify_client.get_recommendations(
            seed_tracks=track_seeds,
            seed_artists=artist_seeds,
            seed_genres=seed_genre_list,
            limit=request_limit,
            tunable_params=tunable_params,
        )
        candidate_payloads.extend(rec_payload.get("tracks") or [])
    except SpotifyClientError as exc:
        logger.warning("Spotify recommendations failed for %s: %s", seed_track.id, exc)

    desired_pool_size = max(limit * 2, limit + 3)
    unique_candidate_ids: Set[str] = {
        payload.get("id")
        for payload in candidate_payloads
        if isinstance(payload, dict) and payload.get("id")
    }

    # Fetch top tracks from seed artists (limited to avoid dominating results)
    for artist_id in artist_seeds[:3]:
        try:
            top_tracks = await spotify_client.get_artist_top_tracks(artist_id, market="US")
        except SpotifyClientError as exc:
            logger.debug("Top tracks fetch failed for %s: %s", artist_id, exc)
            continue
        for item in (top_tracks.get("tracks") or [])[:5]:  # Reduced from 10 to 5
            track_id = item.get("id") if isinstance(item, dict) else None
            if not track_id or track_id in unique_candidate_ids or track_id in exclude_ids:
                continue
            candidate_payloads.append(item)
            unique_candidate_ids.add(track_id)

    # ALWAYS fetch from related artists for diversity (not just when pool is small)
    logger.info(f"Fetching related artists for discovery from {len(artist_seeds[:3])} seed artists")
    for artist_id in artist_seeds[:3]:
        try:
            related_payload = await spotify_client.get_related_artists(artist_id)
        except SpotifyClientError as exc:
            logger.debug("Related artists fetch failed for %s: %s", artist_id, exc)
            continue
        related_artists = (related_payload.get("artists") or [])[:8]  # Increased from 5 to 8
        logger.info(f"Found {len(related_artists)} related artists for {artist_id}")
        for related_artist in related_artists:
            if not isinstance(related_artist, dict):
                continue
            related_artist_id = related_artist.get("id")
            if not related_artist_id:
                continue
            try:
                top_tracks = await spotify_client.get_artist_top_tracks(related_artist_id, market="US")
            except SpotifyClientError:
                continue
            for item in (top_tracks.get("tracks") or [])[:5]:  # Increased from 3 to 5
                track_id = item.get("id") if isinstance(item, dict) else None
                if not track_id or track_id in unique_candidate_ids or track_id in exclude_ids:
                    continue
                candidate_payloads.append(item)
                unique_candidate_ids.add(track_id)
                if len(unique_candidate_ids) >= 100:  # Cap total candidates
                    break
            if len(unique_candidate_ids) >= 100:
                break

    if len(unique_candidate_ids) < desired_pool_size and seed_genre_list:
        for genre in seed_genre_list[:3]:
            try:
                search_results = await spotify_client.search_tracks(f"genre:{genre}", limit=25)
            except SpotifyClientError as exc:
                logger.debug("Genre search failed for %s: %s", genre, exc)
                continue
            tracks = ((search_results.get("tracks") or {}).get("items") or [])
            for item in tracks:
                track_id = item.get("id") if isinstance(item, dict) else None
                if not track_id or track_id in unique_candidate_ids or track_id in exclude_ids:
                    continue
                candidate_payloads.append(item)
                unique_candidate_ids.add(track_id)

    unique_candidates: List[Dict[str, Any]] = []
    seen_track_ids: Set[str] = set()
    for payload in candidate_payloads:
        if not isinstance(payload, dict):
            continue
        track_id = payload.get("id")
        if not track_id or track_id in seen_track_ids or track_id in exclude_ids:
            continue
        seen_track_ids.add(track_id)
        unique_candidates.append(payload)

    if not unique_candidates:
        logger.warning("No candidate tracks found for %s", seed_track.id)
        return []

    logger.info("Evaluating %s candidate tracks for %s", len(unique_candidates), seed_track.id)

    seed_gender = _infer_artist_gender(seed_track.artists)
    seed_artist_ids = {artist.id for artist in seed_track.artists if artist.id}
    seed_popularity = seed_track.popularity

    track_lookup: Dict[str, models.Track] = {}
    scored_tracks: List[Tuple[float, Dict[str, Any], models.Track, Dict[str, Any], SharedGenres]] = []

    for payload in unique_candidates[:120]:
        track_id = payload.get("id")
        if not track_id:
            continue
        try:
            track_obj, _vector, candidate_features, _ = await _ensure_track_vector(
                session,
                spotify_client,
                index_service,
                settings,
                track_payload=payload,
                update_index=False,
            )
        except SpotifyClientError as exc:
            logger.debug("Failed to ingest candidate %s: %s", track_id, exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unexpected error processing candidate %s: %s", track_id, exc)
            continue

        track_with_relations = await _load_track_with_artists(session, track_obj.id)
        if track_with_relations is not None:
            track_obj = track_with_relations

        candidate_gender = _infer_artist_gender(track_obj.artists)
        if seed_gender and candidate_gender and seed_gender != candidate_gender:
            continue

        track_artist_ids = {artist.id for artist in track_obj.artists if artist.id}

        if seed_popularity is not None and track_obj.popularity is not None:
            if abs(seed_popularity - track_obj.popularity) > 35:
                continue

        candidate_genres = _collect_genres(track_obj.artists)

        alignment = _compute_feature_alignment(seed_features, candidate_features)
        genre_overlap = len(seed_genres & candidate_genres)

        seed_instrumental = _coerce_float(seed_features.get("instrumentalness", 0))
        cand_instrumental = _coerce_float(candidate_features.get("instrumentalness", 0))
        seed_speechiness = _coerce_float(seed_features.get("speechiness", 0))
        cand_speechiness = _coerce_float(candidate_features.get("speechiness", 0))
        seed_energy = _coerce_float(seed_features.get("energy", 0.5))
        cand_energy = _coerce_float(candidate_features.get("energy", 0.5))
        seed_valence = _coerce_float(seed_features.get("valence", 0.5))
        cand_valence = _coerce_float(candidate_features.get("valence", 0.5))
        seed_danceability = _coerce_float(seed_features.get("danceability", 0.5))
        cand_danceability = _coerce_float(candidate_features.get("danceability", 0.5))
        seed_loudness = _coerce_float(seed_features.get("loudness", -10))
        cand_loudness = _coerce_float(candidate_features.get("loudness", -10))
        seed_tempo = _coerce_float(seed_features.get("tempo", 120))
        cand_tempo = _coerce_float(candidate_features.get("tempo", 120))

        if seed_instrumental is not None and cand_instrumental is not None and seed_instrumental < 0.5 and cand_instrumental > 0.5:
            continue

        seed_has_vocals = (seed_instrumental or 0) < 0.5 and (seed_speechiness or 0) > 0.05
        cand_has_vocals = (cand_instrumental or 0) < 0.5 and (cand_speechiness or 0) > 0.05
        if seed_has_vocals and not cand_has_vocals:
            continue

        if seed_energy is not None and cand_energy is not None and abs(seed_energy - cand_energy) > 0.18:
            continue
        if seed_valence is not None and cand_valence is not None and abs(seed_valence - cand_valence) > 0.28:
            continue
        if seed_loudness is not None and cand_loudness is not None and abs(seed_loudness - cand_loudness) > 5.5:
            continue
        if seed_tempo is not None and cand_tempo is not None and abs(seed_tempo - cand_tempo) > 15:
            continue
        if seed_danceability is not None and cand_danceability is not None and abs(seed_danceability - cand_danceability) > 0.28:
            continue
        if seed_speechiness is not None and cand_speechiness is not None and abs(seed_speechiness - cand_speechiness) > 0.15:
            continue

        tempo_diff = abs((seed_tempo or 0) - (cand_tempo or 0))
        loudness_diff = abs((seed_loudness or 0) - (cand_loudness or 0))
        energy_diff = abs((seed_energy or 0) - (cand_energy or 0))
        valence_diff = abs((seed_valence or 0) - (cand_valence or 0))
        dance_diff = abs((seed_danceability or 0) - (cand_danceability or 0))
        speech_diff = abs((seed_speechiness or 0) - (cand_speechiness or 0))

        # Calculate vibe score with properly balanced weights (sum to ~0.85 max before bonuses)
        vibe_score = 0.0
        if seed_tempo is not None and cand_tempo is not None:
            vibe_score += max(1 - (tempo_diff / 15), 0) * 0.25  # Tempo is important
        if seed_loudness is not None and cand_loudness is not None:
            vibe_score += max(1 - (loudness_diff / 5.5), 0) * 0.15  # Loudness matters
        vibe_score += max(1 - (energy_diff / 0.18), 0) * 0.20  # Energy is critical
        vibe_score += max(1 - (valence_diff / 0.28), 0) * 0.15  # Mood/valence
        vibe_score += max(1 - (speech_diff / 0.15), 0) * 0.05  # Speechiness
        vibe_score += max(1 - (dance_diff / 0.28), 0) * 0.10  # Danceability

        # Genre overlap bonus (up to 0.10)
        if genre_overlap > 0:
            genre_bonus = min(0.04 + (genre_overlap - 1) * 0.02, 0.10)
            vibe_score += genre_bonus

        # Artist diversity bonus - prioritize new artists
        is_same_artist = bool(track_artist_ids & seed_artist_ids)
        if not is_same_artist:
            vibe_score += 0.05  # Small bonus for discovering new artists

        # Lower threshold to allow more variety while maintaining quality
        if vibe_score < 0.55:
            continue

        track_lookup[track_obj.id] = track_obj
        scored_tracks.append((vibe_score, payload, track_obj, candidate_features, candidate_genres))

    if not scored_tracks:
        logger.warning("After filtering no candidates remain for %s", seed_track.id)
        return []

    scored_tracks.sort(key=lambda item: item[0], reverse=True)

    other_artist_tracks: List[Tuple[float, Dict[str, Any], models.Track, Dict[str, Any], SharedGenres]] = []
    seed_artist_tracks: List[Tuple[float, Dict[str, Any], models.Track, Dict[str, Any], SharedGenres]] = []

    for record in scored_tracks:
        artist_ids = {artist.id for artist in record[2].artists if artist.id}
        if artist_ids & seed_artist_ids:
            seed_artist_tracks.append(record)
        else:
            other_artist_tracks.append(record)

    recommendations: List[RecommendationItem] = []
    added_track_ids: Set[str] = set()
    artist_track_count: Dict[frozenset[str], int] = {}
    seen_artists: Set[str] = set()
    max_per_artist = 2

    def add_record(
        record: Tuple[float, Dict[str, Any], models.Track, Dict[str, Any], SharedGenres]
    ) -> bool:
        vibe_score, _payload, track_obj, candidate_features, candidate_genres = record
        if track_obj.id in added_track_ids:
            return False
        artist_ids = {artist.id for artist in track_obj.artists if artist.id}
        artist_key = frozenset(artist_ids) if artist_ids else frozenset({track_obj.id})
        if artist_track_count.get(artist_key, 0) >= max_per_artist:
            return False
        shared = seed_genres & candidate_genres
        explanation = _build_explanation(seed_features, candidate_features, shared)
        recommendations.append(_track_to_recommendation(track_obj, vibe_score, explanation))
        added_track_ids.add(track_obj.id)
        artist_track_count[artist_key] = artist_track_count.get(artist_key, 0) + 1
        seen_artists.update(artist_ids)
        return True

    # Prioritize discovery: 70% other artists, 30% seed artist
    if limit <= 1:
        desired_other = min(len(other_artist_tracks), limit)
        desired_seed = 0
    elif limit <= 3:
        desired_other = min(len(other_artist_tracks), max(2, limit - 1))
        desired_seed = min(len(seed_artist_tracks), max(0, limit - desired_other))
    else:
        # For larger limits, prioritize discovery (70/30 split)
        target_other = max(1, int(round(limit * 0.70)))
        target_seed = limit - target_other
        desired_other = min(len(other_artist_tracks), target_other)
        desired_seed = min(len(seed_artist_tracks), target_seed)

    # Interleave seed and other artist tracks for variety
    seed_idx = 0
    other_idx = 0
    seed_count_added = 0
    other_count_added = 0

    while len(recommendations) < limit and (seed_idx < len(seed_artist_tracks) or other_idx < len(other_artist_tracks)):
        # Add from seed artist if we haven't hit the desired count
        if seed_idx < len(seed_artist_tracks) and seed_count_added < desired_seed:
            if add_record(seed_artist_tracks[seed_idx]):
                seed_count_added += 1
            seed_idx += 1

        # Add from other artists if we haven't hit the desired count
        if other_idx < len(other_artist_tracks) and other_count_added < desired_other:
            if add_record(other_artist_tracks[other_idx]):
                other_count_added += 1
            other_idx += 1

        # If we've hit our targets but still have slots, keep going
        if seed_count_added >= desired_seed and other_count_added >= desired_other:
            # Fill remaining slots from whichever list has more left
            if seed_idx < len(seed_artist_tracks):
                if add_record(seed_artist_tracks[seed_idx]):
                    seed_count_added += 1
                seed_idx += 1
            elif other_idx < len(other_artist_tracks):
                if add_record(other_artist_tracks[other_idx]):
                    other_count_added += 1
                other_idx += 1
            else:
                break

    other_count = 0
    seed_count = 0
    for item in recommendations:
        track_obj = track_lookup.get(item.track_id)
        if not track_obj:
            continue
        artist_ids = {artist.id for artist in track_obj.artists if artist.id}
        if artist_ids & seed_artist_ids:
            seed_count += 1
        else:
            other_count += 1

    logger.info(
        "Returning %s recommendations (%s other artists, %s seed artist) for %s (from %s scored candidates)",
        len(recommendations),
        other_count,
        seed_count,
        seed_track.id,
        len(scored_tracks),
    )

    # Log sample recommendations for debugging
    if recommendations:
        logger.info("Sample recommendations: %s", [(rec.name, f"{rec.similarity:.2f}") for rec in recommendations[:3]])

    return recommendations

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
        artist_counter: Counter[str] = Counter()
        track_objects: Dict[str, models.Track] = {}
        new_vectors: List[Tuple[str, np.ndarray]] = []
        primary_track: Optional[models.Track] = None
        primary_vector: Optional[np.ndarray] = None
        primary_features: Optional[Dict[str, Any]] = None

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
            artist_counter.update([artist.id for artist in track_obj.artists if artist.id])
            track_objects[track_obj.id] = track_obj
            if updated:
                new_vectors.append((track_obj.id, vector))
            if primary_track is None:
                primary_track = track_obj
                primary_vector = vector
                primary_features = features

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

        # Analyze playlist vibe: compute average audio features
        seed_features_avg = _average_audio_features(feature_payloads)
        seed_genres = set(genre for genre, _ in seed_genres_counter.most_common(20))
        seed_genre_priority = [genre for genre, _ in seed_genres_counter.most_common(5)]

        logger.info(f"Playlist analysis: {len(track_ids)} tracks, top genres: {seed_genre_priority[:3]}")
        logger.info(f"Playlist vibe: tempo={seed_features_avg.get('tempo', 0):.1f}, energy={seed_features_avg.get('energy', 0):.2f}, danceability={seed_features_avg.get('danceability', 0):.2f}")

        recommendations: List[RecommendationItem] = []

        if primary_track is not None:
            # Use average features from entire playlist for better vibe matching
            seed_features_payload = seed_features_avg or dict(primary_features or {})

            # Don't use specific tracks as seeds - we want fresh recommendations based on vibe
            # Get artists from the most common ones in the playlist
            dominant_artist_ids = [artist_id for artist_id, _ in artist_counter.most_common(10) if artist_id]

            logger.info(f"Using {len(dominant_artist_ids)} dominant artists from playlist")

            # Use our custom recommendation function with playlist vibe
            recommendations = await _get_spotify_recommendations(
                session,
                spotify_client,
                seed_track=primary_track,
                seed_features=seed_features_payload,  # Uses AVERAGE features from playlist
                seed_genres=seed_genres,
                seed_genre_priority=seed_genre_priority,
                limit=top_k,
                settings=settings,
                index_service=index_service,
                additional_exclude=set(track_ids),  # Exclude all tracks already in playlist
                extra_seed_tracks=None,  # Don't seed specific tracks
                extra_seed_artists=dominant_artist_ids,  # Use dominant artists
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

    if entity.kind == "track":
        track_payload = await spotify_client.get_track(entity.id)
        track, _vector, seed_features, _ = await _ensure_track_vector(
            session,
            spotify_client,
            index_service,
            settings,
            track_payload=track_payload,
            update_index=False,
        )

        track_with_relations = await _load_track_with_artists(session, track.id)
        if track_with_relations is not None:
            track = track_with_relations

        seed_genre_counter: Counter[str] = Counter()
        for artist in track.artists:
            if artist.genres:
                normalized_genres = [genre.lower() for genre in artist.genres if genre]
                seed_genre_counter.update(normalized_genres)
        seed_genres = set(seed_genre_counter.keys())
        seed_genre_priority = [genre for genre, _ in seed_genre_counter.most_common(5)]

        recommendations = await _get_spotify_recommendations(
            session,
            spotify_client,
            seed_track=track,
            seed_features=seed_features,
            seed_genres=seed_genres,
            seed_genre_priority=seed_genre_priority,
            limit=limit,
            settings=settings,
            index_service=index_service,
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


