from __future__ import annotations

import io
from typing import Any, Dict

import httpx
import librosa
import numpy as np
from sklearn.preprocessing import normalize


def _safe_get(mapping: Dict[str, Any] | None, *path: str, default: float = 0.0) -> float:
    current: Any = mapping or {}
    for step in path:
        if not isinstance(current, dict):
            return default
        current = current.get(step)
        if current is None:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def build_feature_vector(
    audio_features: Dict[str, Any],
    audio_analysis: Dict[str, Any] | None,
    dsp_features: Dict[str, Any] | None,
) -> np.ndarray:
    af = audio_features or {}
    analysis = audio_analysis or {}
    dsp = dsp_features or {}

    values = [
        float(af.get("danceability", 0.0)),
        float(af.get("energy", 0.0)),
        float(af.get("speechiness", 0.0)),
        float(af.get("acousticness", 0.0)),
        float(af.get("instrumentalness", 0.0)),
        float(af.get("liveness", 0.0)),
        float(af.get("valence", 0.0)),
        float(af.get("tempo", 0.0)) / 250.0,
        float(af.get("key", -1.0)) / 11.0,
        float(af.get("mode", 0.0)),
        float(af.get("loudness", -60.0)) / 60.0,
        float(af.get("duration_ms", 0.0)) / (1000.0 * 60.0 * 10.0),
    ]

    track_meta = analysis.get("track") if isinstance(analysis, dict) else {}
    sections = analysis.get("sections") if isinstance(analysis, dict) else None

    values.extend(
        [
            _safe_get(track_meta, "tempo_confidence", default=0.0),
            _safe_get(track_meta, "time_signature", default=4.0) / 7.0,
            _safe_get(track_meta, "time_signature_confidence", default=0.0),
            _safe_get(track_meta, "key_confidence", default=0.0),
            float(len(sections) if isinstance(sections, list) else 0) / 20.0,
        ]
    )

    if dsp:
        values.extend(
            [
                float(dsp.get("spectral_centroid_mean", 0.0)) / 4000.0,
                float(dsp.get("spectral_bandwidth_mean", 0.0)) / 4000.0,
                float(dsp.get("spectral_contrast_mean", 0.0)) / 40.0,
                float(dsp.get("rolloff_mean", 0.0)) / 8000.0,
                float(dsp.get("zero_crossing_rate", 0.0)),
            ]
        )

    vector = np.array(values, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    vector = normalize(vector.reshape(1, -1))[0]
    return vector.astype(np.float32)


async def compute_dsp_features(preview_url: str | None, *, timeout: float = 10.0) -> Dict[str, float]:
    if not preview_url:
        return {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(preview_url)
            response.raise_for_status()
            audio_bytes = response.content
    except Exception:
        return {}

    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        if y.size == 0:
            return {}
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_contrast_mean": float(np.mean(spectral_contrast)),
            "rolloff_mean": float(np.mean(rolloff)),
            "zero_crossing_rate": float(np.mean(zcr)),
        }
    except Exception:
        return {}