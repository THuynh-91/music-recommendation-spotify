from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import Settings
from ...core.security import verify_service_token
from ...schemas.recommend import RecommendRequest, RecommendResponse
from ...services.index import FaissService
from ...services.recommendations import recommend_for_entity
from ...spotify.client import SpotifyAuthError, SpotifyClientError
from ...spotify.parsing import parse_spotify_url
from ..deps import (
    get_db_session,
    get_index_service,
    get_redis_dep,
    get_settings_dep,
    get_spotify_client,
)

router = APIRouter(prefix="/v1", tags=["recommendations"], dependencies=[Depends(verify_service_token)])


@router.post("/recommendations", response_model=RecommendResponse)
async def create_recommendations(
    payload: RecommendRequest,
    *,
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_dep),
    spotify_client = Depends(get_spotify_client),
    index_service: FaissService = Depends(get_index_service),
    settings: Settings = Depends(get_settings_dep),
) -> RecommendResponse:
    entity = parse_spotify_url(payload.url)
    try:
        limit = payload.limit or settings.recommendation_top_k
        limit = max(1, min(limit, settings.recommendation_max_limit))

        response = await recommend_for_entity(
            entity,
            session=session,
            redis=redis,
            spotify_client=spotify_client,
            index_service=index_service,
            settings=settings,
            limit=limit,
        )
        return response
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except SpotifyClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc