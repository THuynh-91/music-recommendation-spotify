from __future__ import annotations

from collections.abc import AsyncIterator

from redis.asyncio import Redis

from ..core.config import get_settings

settings = get_settings()

redis = Redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)


async def get_redis() -> AsyncIterator[Redis]:
    try:
        yield redis
    finally:
        # keep connection open for reuse; do not close
        pass


async def acquire_lock(key: str, *, ttl: int = 60) -> bool:
    return await redis.set(name=key, value="1", nx=True, ex=ttl)


async def release_lock(key: str) -> None:
    await redis.delete(key)