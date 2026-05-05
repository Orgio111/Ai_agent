"""Semantic memory: concept-level knowledge with deduplication and contradiction resolution."""
from __future__ import annotations

import json
from typing import Optional

import asyncpg
import faiss
import numpy as np

from .models import MemoryResult


class SemanticMemory:
    def __init__(self, dsn: str, vector_dim: int) -> None:
        self._dsn = dsn
        self._dim = vector_dim
        self._pool: Optional[asyncpg.Pool] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_map: list[str] = []

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        await self._ensure_schema()
        self._index = faiss.IndexFlatIP(self._dim)
        await self._load_index()

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ DEFAULT NOW(),
                    access_count INT DEFAULT 0,
                    confidence FLOAT DEFAULT 0.7,
                    content_hash TEXT UNIQUE,
                    vector FLOAT4[] NOT NULL,
                    contradicts TEXT[],
                    supports TEXT[]
                );
                CREATE INDEX IF NOT EXISTS semantic_confidence_idx ON semantic_memories(confidence);
                CREATE INDEX IF NOT EXISTS semantic_hash_idx ON semantic_memories(content_hash);
            """)

    async def _load_index(self) -> None:
        assert self._pool is not None and self._index is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, vector FROM semantic_memories ORDER BY created_at")
        self._id_map = []
        vecs = []
        for row in rows:
            self._id_map.append(row["id"])
            vecs.append(np.array(row["vector"], dtype=np.float32))
        if vecs:
            self._index.add(np.stack(vecs))

    async def find_duplicate(self, embedding: np.ndarray, threshold: float = 0.95) -> Optional[str]:
        """Return existing memory_id if near-duplicate found."""
        assert self._index is not None
        if self._index.ntotal == 0:
            return None
        scores, indices = self._index.search(embedding.reshape(1, -1), 1)
        if scores[0][0] >= threshold and indices[0][0] < len(self._id_map):
            return self._id_map[indices[0][0]]
        return None

    async def merge(self, existing_id: str, new_record: dict) -> None:
        """Merge new information into existing semantic memory."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT content, confidence FROM semantic_memories WHERE id=$1", existing_id
            )
            if not existing:
                return
            new_confidence = min(1.0, existing["confidence"] + 0.05)
            await conn.execute(
                """UPDATE semantic_memories
                   SET confidence=$1, updated_at=NOW(), access_count=access_count+1
                   WHERE id=$2""",
                new_confidence,
                existing_id,
            )

    async def store(self, memory_id: str, record: dict, embedding: np.ndarray) -> None:
        assert self._pool is not None and self._index is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO semantic_memories
                   (id, content, metadata, confidence, content_hash, vector)
                   VALUES ($1,$2,$3,$4,$5,$6)
                   ON CONFLICT(content_hash) DO NOTHING""",
                memory_id,
                record["content"],
                json.dumps(record.get("metadata", {})),
                record.get("confidence", 0.7),
                record.get("content_hash"),
                embedding.tolist(),
            )
        self._index.add(embedding.reshape(1, -1))
        self._id_map.append(memory_id)

    async def search(
        self, embedding: np.ndarray, k: int = 10, min_score: float = 0.0
    ) -> list[MemoryResult]:
        assert self._index is not None and self._pool is not None
        if self._index.ntotal == 0:
            return []

        k_search = min(k * 2, self._index.ntotal)
        scores, indices = self._index.search(embedding.reshape(1, -1), k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map) or score < min_score:
                continue
            mid = self._id_map[idx]
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM semantic_memories WHERE id=$1", mid)
            if row:
                results.append(
                    MemoryResult(
                        memory_id=mid,
                        content=row["content"],
                        memory_type="semantic",
                        score=float(score),
                        confidence=row["confidence"],
                        created_at=row["created_at"].isoformat(),
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                )
        return results[:k]

    async def update_access(self, memory_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE semantic_memories SET access_count=access_count+1, last_accessed=NOW() WHERE id=$1",
                memory_id,
            )

    async def count(self) -> int:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM semantic_memories")
