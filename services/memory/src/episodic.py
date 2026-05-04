"""Episodic memory: PostgreSQL metadata + FAISS vector index."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import faiss
import numpy as np

from .models import MemoryResult


class EpisodicMemory:
    def __init__(self, dsn: str, vector_dim: int) -> None:
        self._dsn = dsn
        self._dim = vector_dim
        self._pool: Optional[asyncpg.Pool] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_map: list[str] = []  # FAISS idx → memory_id

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        await self._ensure_schema()
        self._index = faiss.IndexFlatIP(self._dim)
        await self._load_index()

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    session_id TEXT,
                    agent_id TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ DEFAULT NOW(),
                    access_count INT DEFAULT 0,
                    importance FLOAT DEFAULT 0.5,
                    confidence FLOAT DEFAULT 0.5,
                    content_hash TEXT,
                    vector FLOAT4[] NOT NULL,
                    decayed BOOLEAN DEFAULT FALSE
                );
                CREATE INDEX IF NOT EXISTS episodic_session_idx ON episodic_memories(session_id);
                CREATE INDEX IF NOT EXISTS episodic_created_idx ON episodic_memories(created_at);
                CREATE INDEX IF NOT EXISTS episodic_confidence_idx ON episodic_memories(confidence);
            """)

    async def _load_index(self) -> None:
        assert self._pool is not None and self._index is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, vector FROM episodic_memories WHERE NOT decayed ORDER BY created_at"
            )
        self._id_map = []
        vecs = []
        for row in rows:
            self._id_map.append(row["id"])
            vecs.append(np.array(row["vector"], dtype=np.float32))
        if vecs:
            mat = np.stack(vecs)
            self._index.add(mat)

    async def store(self, memory_id: str, record: dict, embedding: np.ndarray) -> None:
        assert self._pool is not None and self._index is not None
        vec_list = embedding.tolist()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodic_memories
                    (id, content, session_id, agent_id, metadata, importance, confidence, content_hash, vector)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT(id) DO NOTHING
                """,
                memory_id,
                record["content"],
                record.get("session_id"),
                record.get("agent_id"),
                json.dumps(record.get("metadata", {})),
                record.get("importance", 0.5),
                record.get("confidence", 0.5),
                record.get("content_hash"),
                vec_list,
            )
        self._index.add(embedding.reshape(1, -1))
        self._id_map.append(memory_id)

    async def search(
        self,
        embedding: np.ndarray,
        k: int = 10,
        session_id: Optional[str] = None,
        max_age_hours: Optional[float] = None,
        min_score: float = 0.0,
    ) -> list[MemoryResult]:
        assert self._index is not None and self._pool is not None
        if self._index.ntotal == 0:
            return []

        k_search = min(k * 3, self._index.ntotal)
        scores, indices = self._index.search(embedding.reshape(1, -1), k_search)

        candidate_ids = []
        candidate_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            mid = self._id_map[idx]
            if score >= min_score:
                candidate_ids.append(mid)
                candidate_scores[mid] = float(score)

        if not candidate_ids:
            return []

        conditions = ["id = ANY($1)", "NOT decayed"]
        params: list[Any] = [candidate_ids]
        p = 2

        if session_id:
            conditions.append(f"(session_id = ${p} OR session_id IS NULL)")
            params.append(session_id)
            p += 1

        if max_age_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            conditions.append(f"created_at >= ${p}")
            params.append(cutoff)
            p += 1

        sql = f"SELECT * FROM episodic_memories WHERE {' AND '.join(conditions)} LIMIT {k * 2}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        now = datetime.now(timezone.utc)
        for row in rows:
            mid = row["id"]
            age_hours = (now - row["created_at"].replace(tzinfo=timezone.utc)).total_seconds() / 3600
            results.append(
                MemoryResult(
                    memory_id=mid,
                    content=row["content"],
                    memory_type="episodic",
                    score=candidate_scores.get(mid, 0.0),
                    confidence=row["confidence"],
                    created_at=row["created_at"].isoformat(),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    age_hours=age_hours,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def delete(self, memory_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("UPDATE episodic_memories SET decayed=TRUE WHERE id=$1", memory_id)

    async def delete_older_than(self, hours: float) -> int:
        assert self._pool is not None
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE episodic_memories SET decayed=TRUE WHERE created_at < $1 AND NOT decayed",
                cutoff,
            )
        return int(result.split()[-1])

    async def update_access(self, memory_id: str) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE episodic_memories SET access_count=access_count+1, last_accessed=NOW() WHERE id=$1",
                memory_id,
            )

    async def get_low_confidence(self, threshold: float = 0.2) -> list[str]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM episodic_memories WHERE confidence < $1 AND NOT decayed",
                threshold,
            )
        return [r["id"] for r in rows]

    async def count(self) -> int:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM episodic_memories WHERE NOT decayed")

    async def ping(self) -> bool:
        try:
            assert self._pool is not None
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
