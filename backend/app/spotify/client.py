from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Sequence

import httpx

from ..core.config import Settings

API_BASE = "https://api.spotify.com/v1"
TOKEN_ENDPOINT = "https://accounts.spotify.com/api/token"


class SpotifyClientError(Exception):
    pass


class SpotifyAuthError(SpotifyClientError):
    pass


@dataclass(slots=True)
class SpotifyClient:
    access_token: str
    timeout: float = 15.0
    retries: int = 3
    settings: Settings | None = None
    _client: httpx.AsyncClient | None = field(init=False, repr=False, default=None)
    _app_token: tuple[str, float] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_app_token(self) -> str:
        if not self.settings:
            raise SpotifyClientError("application settings not available for client credentials")
        client_id = self.settings.spotify_client_id
        client_secret = self.settings.spotify_client_secret
        if not client_id or not client_secret:
            raise SpotifyClientError("missing spotify client credentials")

        if self._app_token and self._app_token[1] > time.time():
            return self._app_token[0]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                TOKEN_ENDPOINT,
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
            )
        data = resp.json()
        if resp.status_code != 200 or "access_token" not in data:
            raise SpotifyClientError(f"failed to obtain client credentials token: {resp.status_code} {data}")
        expires = time.time() + float(data.get("expires_in", 3600)) - 30
        token = data["access_token"]
        self._app_token = (token, expires)
        return token

    async def _request_with_app_token(
        self,
        method: str,
        url: str,
        *,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        token = await self._ensure_app_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        endpoint = url
        query_params = params or {}
        single_audio_feature = False

        if url.startswith("/audio-features/"):
            track_id = url.rsplit("/", 1)[-1]
            endpoint = "/audio-features"
            query_params = {"ids": track_id}
            single_audio_feature = True

        async with httpx.AsyncClient(base_url=API_BASE, headers=headers, timeout=self.timeout) as client:
            resp = await client.request(method, endpoint, params=query_params)

        if resp.status_code >= 400:
            raise SpotifyClientError(f"spotify app-token api error {resp.status_code}: {resp.text}")
        if not resp.content:
            return {}
        payload = resp.json()
        if single_audio_feature:
            features = payload.get("audio_features") or []
            return features[0] if features else {}
        return payload

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        client = self._client
        if client is None:
            raise SpotifyClientError("spotify client not initialized")

        logger = logging.getLogger("spotify.client")

        # Strip leading slash to avoid double slashes with base_url
        clean_url = url.lstrip("/")

        for attempt in range(1, self.retries + 1):
            try:
                if "recommendations" in clean_url:
                    logger.info(f"Recommendations API full URL: {client.base_url}/{clean_url}")
                    logger.info(f"Recommendations API params: {params}")
                response = await client.request(method, clean_url, params=params, json=json)
                if url == "/recommendations":
                    logger.info(f"Recommendations API response status: {response.status_code}")
                    logger.info(f"Recommendations API full request URL: {response.request.url}")
                    logger.info(f"Recommendations API response body: {response.text[:500]}")
            except httpx.RequestError as exc:  # network issue
                if attempt == self.retries:
                    raise SpotifyClientError(f"network error: {exc}") from exc
                await asyncio.sleep(2 ** attempt)
                continue

            if response.status_code == 401:
                raise SpotifyAuthError("spotify token unauthorized")

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "1"))
                await asyncio.sleep(retry_after)
                continue

            # Don't use client credentials fallback for recommendations - it requires user token
            if response.status_code == 403 and method == "GET" and url != "/recommendations" and any(
                url.startswith(path) for path in ("/audio-features", "/audio-analysis")
            ):
                try:
                    return await self._request_with_app_token(method, url, params=params)
                except SpotifyClientError as app_exc:
                    logger.error("Client credentials fallback failed for %s %s: %s", method, url, app_exc)
                    raise

            if response.status_code >= 400:
                detail = response.text
                logger.error("Spotify API %s %s -> %s %s", method, url, response.status_code, detail)
                raise SpotifyClientError(f"spotify api error {response.status_code}: {detail}")

            if response.content:
                return response.json()
            return {}

        raise SpotifyClientError("max retries exceeded for spotify request")

    async def get_track(self, track_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/tracks/{track_id}")

    async def get_tracks(self, track_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(track_ids)
        out: List[Dict[str, Any]] = []
        for chunk_start in range(0, len(ids), 50):
            chunk = ids[chunk_start: chunk_start + 50]
            payload = await self._request("GET", "/tracks", params={"ids": ",".join(chunk)})
            out.extend(payload.get("tracks", []))
        return out

    async def get_artists(self, artist_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list({a for a in artist_ids if a})
        artists: List[Dict[str, Any]] = []
        for start in range(0, len(ids), 50):
            chunk = ids[start: start + 50]
            data = await self._request("GET", "/artists", params={"ids": ",".join(chunk)})
            artists.extend(data.get("artists", []))
        return artists

    async def get_related_artists(self, artist_id: str) -> Dict[str, Any]:
        if not artist_id:
            return {"artists": []}
        return await self._request("GET", f"/artists/{artist_id}/related-artists")

    async def get_artist_top_tracks(self, artist_id: str, market: str = "US") -> Dict[str, Any]:
        if not artist_id:
            return {"tracks": []}
        return await self._request("GET", f"/artists/{artist_id}/top-tracks", params={"market": market})

    async def get_audio_features(self, track_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/audio-features/{track_id}")

    async def get_audio_features_bulk(self, track_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(track_ids)
        features: List[Dict[str, Any]] = []
        for start in range(0, len(ids), 100):
            chunk = ids[start: start + 100]
            data = await self._request("GET", "/audio-features", params={"ids": ",".join(chunk)})
            features.extend(data.get("audio_features", []))
        return features

    async def get_audio_analysis(self, track_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/audio-analysis/{track_id}")

    async def get_recommendations(
        self,
        *,
        seed_tracks: Sequence[str] | None = None,
        seed_artists: Sequence[str] | None = None,
        seed_genres: Sequence[str] | None = None,
        limit: int = 20,
        tunable_params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        logger = logging.getLogger("spotify.client")

        params: Dict[str, Any] = {"limit": max(1, min(limit, 100))}

        # Clean and validate seeds
        clean_tracks = list(dict.fromkeys([track for track in (seed_tracks or []) if track]))[:5]
        clean_artists = list(dict.fromkeys([artist for artist in (seed_artists or []) if artist]))[:5]
        clean_genres = list(dict.fromkeys([genre for genre in (seed_genres or []) if genre]))[:5]

        # Must have at least one seed
        total_seeds = len(clean_tracks) + len(clean_artists) + len(clean_genres)
        if total_seeds == 0:
            raise SpotifyClientError("at least one seed required for recommendations")

        # Maximum 5 seeds total for Spotify API
        if total_seeds > 5:
            # Prioritize tracks, then artists, then genres
            remaining = 5
            final_tracks = clean_tracks[:min(len(clean_tracks), remaining)]
            remaining -= len(final_tracks)
            final_artists = clean_artists[:min(len(clean_artists), remaining)]
            remaining -= len(final_artists)
            final_genres = clean_genres[:min(len(clean_genres), remaining)]

            clean_tracks = final_tracks
            clean_artists = final_artists
            clean_genres = final_genres

        if clean_tracks:
            params["seed_tracks"] = ",".join(clean_tracks)
        if clean_artists:
            params["seed_artists"] = ",".join(clean_artists)
        if clean_genres:
            params["seed_genres"] = ",".join(clean_genres)

        if tunable_params:
            params.update({key: value for key, value in tunable_params.items() if value is not None})

        logger.info(f"Recommendations request: tracks={clean_tracks}, artists={clean_artists}, genres={clean_genres}, limit={limit}")
        logger.debug(f"Full params: {params}")

        try:
            result = await self._request("GET", "/recommendations", params=params)
            logger.info(f"Recommendations returned {len(result.get('tracks', []))} tracks")
            return result
        except SpotifyClientError as exc:
            logger.error(f"Recommendations failed with params: {params}, error: {exc}")
            raise

    async def get_playlist(self, playlist_id: str) -> Dict[str, Any]:
        params = {
            "fields": "id,name,description,images,owner(id,display_name),snapshot_id,tracks.total",
        }
        return await self._request("GET", f"/playlists/{playlist_id}", params=params)

    async def iter_playlist_tracks(self, playlist_id: str, batch_size: int = 100) -> AsyncIterator[Dict[str, Any]]:
        url = f"/playlists/{playlist_id}/tracks"
        params: Dict[str, Any] = {
            "limit": batch_size,
            "fields": "items(added_at,track(id,name,uri,duration_ms,explicit,preview_url,popularity,external_urls,href,album(id,name,images,release_date,release_date_precision),artists(id,name,genres,images,popularity))),next",
        }
        while True:
            data = await self._request("GET", url, params=params)
            for item in data.get("items", []) or []:
                if not item:
                    continue
                yield item
            next_url = data.get("next")
            if not next_url:
                break
            url = next_url
            params = None

    async def search_tracks(self, query: str, *, limit: int = 10) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/search",
            params={"q": query, "type": "track", "limit": limit},
        )