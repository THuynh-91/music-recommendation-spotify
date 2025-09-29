from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...schemas.recommend import HealthResponse
from ..deps import get_db_session, get_index_service
from ...services.index import FaissService

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(
    session: AsyncSession = Depends(get_db_session),
    index_service: FaissService = Depends(get_index_service),
) -> HealthResponse:
    await index_service.ensure_loaded(session)
    size = index_service.index.ntotal if index_service.index is not None else 0
    return HealthResponse(ok=True, index_size=size)