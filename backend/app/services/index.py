from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings, get_settings
from ..db import models


logger = logging.getLogger("faiss")

class FaissService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.index: faiss.IndexIDMap | None = None
        self.id_to_track: Dict[int, str] = {}
        self.track_to_id: Dict[str, int] = {}
        self.dim: int | None = None
        self.lock = asyncio.Lock()
        self.loaded = False

    @staticmethod
    def _track_key(track_id: str) -> int:
        digest = hashlib.blake2b(track_id.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        return value & 0x7FFFFFFFFFFFFFFF

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(quantizer)
            faiss.normalize_L2  # ensure symbol exists
            self.index = index
            self.dim = dim
        elif self.dim != dim:
            raise ValueError(f"FAISS index dimension mismatch (have {self.dim}, expected {dim})")

    async def ensure_loaded(self, session: AsyncSession) -> None:
        async with self.lock:
            if self.loaded:
                return
            index_path = Path(self.settings.faiss_index_path)
            meta_path = Path(self.settings.faiss_meta_path)
            if index_path.exists() and meta_path.exists():
                self.index = faiss.read_index(str(index_path))  # type: ignore[arg-type]
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.id_to_track = {int(k): v for k, v in meta.get("id_to_track", {}).items()}
                self.track_to_id = {v: int(k) for k, v in self.id_to_track.items()}
                self.dim = meta.get("dim")
                self.loaded = True
                return
            await self._rebuild_from_db(session)

    async def _rebuild_from_db(self, session: AsyncSession) -> None:
        result = await session.execute(select(models.TrackFeature.track_id, models.TrackFeature.vector, models.TrackFeature.vector_dim))
        rows = result.all()
        if not rows:
            self.index = None
            self.id_to_track = {}
            self.track_to_id = {}
            self.dim = None
            self.loaded = True
            return

        dim = rows[0].vector_dim
        self._ensure_index(dim)
        ids = []
        vectors = []
        self.id_to_track.clear()
        self.track_to_id.clear()
        for track_id, vector_bytes, vec_dim in rows:
            if vec_dim != dim:
                continue
            vec = np.frombuffer(vector_bytes, dtype=np.float32)
            vec = vec.reshape(1, dim)
            key = self._track_key(track_id)
            ids.append(key)
            vectors.append(vec)
            self.id_to_track[key] = track_id
            self.track_to_id[track_id] = key

        if vectors:
            matrix = np.concatenate(vectors, axis=0)
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            keys = np.array(ids, dtype=np.int64)
            self.index.reset()
            self.index.add_with_ids(matrix, keys)

        self.loaded = True
        await self._persist()

    async def add_vectors(self, items: Iterable[Tuple[str, np.ndarray]]) -> None:
        async with self.lock:
            items_list = list(items)
            if not items_list:
                return
            dim = items_list[0][1].shape[0]
            if self.index is None:
                self._ensure_index(dim)
            elif self.dim != dim:
                # Rebuild from scratch
                raise ValueError("vector dimension mismatch; rebuild FAISS index")

            vectors = []
            ids = []
            for track_id, vector in items_list:
                faiss_id = self._track_key(track_id)
                if track_id in self.track_to_id:
                    old_id = self.track_to_id[track_id]
                    remove_ids = np.array([old_id], dtype=np.int64)
                    self.index.remove_ids(remove_ids)
                vectors.append(vector.reshape(1, -1).astype(np.float32))
                ids.append(faiss_id)
                self.track_to_id[track_id] = faiss_id
                self.id_to_track[faiss_id] = track_id

            matrix = np.concatenate(vectors, axis=0)
            matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            id_array = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(matrix, id_array)
            await self._persist()

    async def search(self, vector: np.ndarray, *, top_k: int) -> List[Tuple[str, float]]:
        async with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            matrix = vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(matrix)
            scores, ids = self.index.search(matrix, top_k)
        matches: List[Tuple[str, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            track_id = self.id_to_track.get(int(idx))
            if track_id:
                matches.append((track_id, float(score)))
        return matches

    async def _persist(self) -> None:
        if self.index is None:
            return
        index_path = Path(self.settings.faiss_index_path)
        meta_path = Path(self.settings.faiss_meta_path)
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))  # type: ignore[arg-type]
            meta = {
                "id_to_track": {str(k): v for k, v in self.id_to_track.items()},
                "dim": self.dim,
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem issues
            logger.warning("Failed to persist FAISS index to %s: %s", index_path, exc)


faiss_service = FaissService()


async def get_faiss_service() -> FaissService:
    return faiss_service

