from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from ..cache.redis import get_redis
from ..core.config import Settings, get_settings
from ..core.security import extract_spotify_access_token
from ..db.session import get_session
from ..services.index import FaissService, get_faiss_service
from ..spotify.client import SpotifyClient


async def get_settings_dep() -> Settings:
    return get_settings()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async for session in get_session():
        yield session


async def get_redis_dep() -> AsyncIterator[Redis]:
    async for client in get_redis():
        yield client


async def get_spotify_client(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> AsyncIterator[SpotifyClient]:
    token = extract_spotify_access_token(request)
    client = SpotifyClient(access_token=token, timeout=settings.http_timeout_seconds, retries=settings.http_retries)
    try:
        yield client
    finally:
        await client.close()


async def get_index_service() -> FaissService:
    return await get_faiss_service()