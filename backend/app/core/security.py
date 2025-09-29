from __future__ import annotations

import secrets
from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


def verify_service_token(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.service_token
    if not expected:
        # If no token configured we allow all traffic (useful for local dev), but log warning once.
        request.app.state.service_token_warning = True  # type: ignore[attr-defined]
        return

    provided = request.headers.get("X-Service-Token", "")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid service token")


def extract_spotify_access_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if auth.startswith(prefix):
        token = auth[len(prefix):].strip()
        if token:
            return token
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing spotify access token")
