"""Procedural memory: stores learned agent procedures and action patterns."""
from __future__ import annotations

import json
from typing import Optional

import asyncpg

from .models import MemoryResult


class ProceduralMemory:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        await self._ensure_schema()

    async def _ensure_schema(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS procedural_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    procedure_name TEXT,
                    success_count INT DEFAULT 0,
                    failure_count INT DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    confidence FLOAT DEFAULT 0.5
                );
                CREATE INDEX IF NOT EXISTS proc_name_idx ON procedural_memories(procedure_name);
                CREATE INDEX IF NOT EXISTS proc_confidence_idx ON procedural_memories(confidence);
            """)

    async def store(self, memory_id: str, record: dict) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO procedural_memories
                   (id, content, procedure_name, metadata, confidence)
                   VALUES ($1,$2,$3,$4,$5)
                   ON CONFLICT(id) DO UPDATE
                   SET content=$2, updated_at=NOW(), confidence=$5""",
                memory_id,
                record["content"],
                record.get("metadata", {}).get("procedure_name"),
                json.dumps(record.get("metadata", {})),
                record.get("confidence", 0.5),
            )

    async def search(self, query: str, limit: int = 10) -> list[MemoryResult]:
        assert self._pool is not None
        # Full-text search via PostgreSQL LIKE (upgrade to pg_trgm for production)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM procedural_memories
                   WHERE content ILIKE $1 OR procedure_name ILIKE $1
                   ORDER BY confidence DESC, success_count DESC
                   LIMIT $2""",
                f"%{query[:100]}%",
                limit,
            )
        return [
            MemoryResult(
                memory_id=row["id"],
                content=row["content"],
                memory_type="procedural",
                score=row["confidence"],
                confidence=row["confidence"],
                created_at=row["created_at"].isoformat(),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    async def record_outcome(self, memory_id: str, success: bool) -> None:
        assert self._pool is not None
        if success:
            field = "success_count"
            delta_confidence = 0.05
        else:
            field = "failure_count"
            delta_confidence = -0.05
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""UPDATE procedural_memories
                    SET {field}={field}+1,
                        confidence=LEAST(1.0, GREATEST(0.0, confidence+$1)),
                        updated_at=NOW()
                    WHERE id=$2""",
                delta_confidence,
                memory_id,
            )

    async def count(self) -> int:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM procedural_memories")
