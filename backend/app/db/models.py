from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, LargeBinary, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


track_artists = Table(
    "track_artists",
    Base.metadata,
    Column("track_id", String(64), ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True),
    Column("artist_id", String(64), ForeignKey("artists.id", ondelete="CASCADE"), primary_key=True),
)


class Artist(Base):
    __tablename__ = "artists"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    genres: Mapped[list[str] | None] = mapped_column(JSONB, default=list)
    popularity: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tracks: Mapped[list["Track"]] = relationship(back_populates="artists", secondary=track_artists)


class Album(Base):
    __tablename__ = "albums"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    release_date: Mapped[str | None] = mapped_column(String(32))
    release_date_precision: Mapped[str | None] = mapped_column(String(16))
    image_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tracks: Mapped[list["Track"]] = relationship(back_populates="album")


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    album_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("albums.id", ondelete="SET NULL"))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    explicit: Mapped[bool] = mapped_column(Boolean, default=False)
    popularity: Mapped[int | None] = mapped_column(Integer)
    preview_url: Mapped[str | None] = mapped_column(Text)
    external_url: Mapped[str | None] = mapped_column(Text)
    uri: Mapped[str | None] = mapped_column(String(128))
    image_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    album: Mapped[Album | None] = relationship(back_populates="tracks", lazy="selectin")
    artists: Mapped[list[Artist]] = relationship(back_populates="tracks", secondary=track_artists, lazy="selectin")
    features: Mapped[TrackFeature | None] = relationship(back_populates="track", uselist=False)


class TrackFeature(Base):
    __tablename__ = "track_features"

    track_id: Mapped[str] = mapped_column(String(64), ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    vector_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    audio_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    audio_analysis: Mapped[dict | None] = mapped_column(JSONB)
    dsp_features: Mapped[dict | None] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    track: Mapped[Track] = relationship(back_populates="features")


class Playlist(Base):
    __tablename__ = "playlists"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[str | None] = mapped_column(String(64))
    owner_display_name: Mapped[str | None] = mapped_column(String(255))
    snapshot_id: Mapped[str | None] = mapped_column(String(128))
    last_ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    image_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tracks: Mapped[list["PlaylistTrack"]] = relationship(back_populates="playlist", cascade="all, delete-orphan")


class PlaylistTrack(Base):
    __tablename__ = "playlist_tracks"

    playlist_id: Mapped[str] = mapped_column(String(64), ForeignKey("playlists.id", ondelete="CASCADE"), primary_key=True)
    track_id: Mapped[str] = mapped_column(String(64), ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    added_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    playlist: Mapped[Playlist] = relationship(back_populates="tracks")
    track: Mapped[Track] = relationship()

