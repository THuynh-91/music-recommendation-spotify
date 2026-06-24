from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import Settings
from ...schemas.recommend import HealthResponse
from ..deps import get_db_session, get_index_service, get_settings_dep
from ...services.index import FaissService

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(
    session: AsyncSession | None = Depends(get_db_session),
    index_service: FaissService | None = Depends(get_index_service),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    if settings.demo_mode or index_service is None or session is None:
        return HealthResponse(ok=True, index_size=0)
    await index_service.ensure_loaded(session)
    size = index_service.index.ntotal if index_service.index is not None else 0
    return HealthResponse(ok=True, index_size=size)