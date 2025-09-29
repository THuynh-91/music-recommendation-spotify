from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..spotify.parsing import parse_spotify_url


class RecommendRequest(BaseModel):
    url: str = Field(..., description="Spotify track or playlist URL/URI")

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        parse_spotify_url(value)
        return value


class SeedTrack(BaseModel):
    id: str
    name: str
    artists: List[str]
    image_url: Optional[str] = None
    external_url: Optional[str] = None


class SeedPlaylist(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    image_url: Optional[str] = None
    track_count: int = 0


class PlaylistIngestSummary(BaseModel):
    id: str
    name: str
    track_count: int
    ingested_tracks: int
    snapshot_id: Optional[str] = None


class RecommendationItem(BaseModel):
    track_id: str
    name: str
    artists: List[str]
    preview_url: Optional[str] = None
    external_url: Optional[str] = None
    image_url: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class RecommendResponse(BaseModel):
    type: Literal["track", "playlist"]
    seed_track: Optional[SeedTrack] = None
    seed_playlist: Optional[SeedPlaylist] = None
    playlist: Optional[PlaylistIngestSummary] = None
    recommendations: List[RecommendationItem] = []


class HealthResponse(BaseModel):
    ok: bool = True
    index_size: int = 0