from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List

import httpx

API_BASE = "https://api.spotify.com/v1"
TOKEN_BASE = "https://accounts.spotify.com"


class SpotifyClientError(Exception):
    pass


class SpotifyAuthError(SpotifyClientError):
    pass


@dataclass(slots=True)
class SpotifyClient:
    access_token: str
    timeout: float = 15.0
    retries: int = 3

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
        base_url: str = API_BASE,
    ) -> Dict[str, Any]:
        for attempt in range(1, self.retries + 1):
            try:
                response = await self._client.request(method, url, params=params, json=json, base_url=base_url)
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

            if response.status_code >= 400:
                detail = response.text
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
