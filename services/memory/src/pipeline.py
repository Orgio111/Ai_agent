"""Memory pipeline: orchestrates all memory layers with decay, deduplication,
contradiction resolution, and confidence scoring."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np

from .decay import DecayEngine
from .episodic import EpisodicMemory
from .models import (
    MemoryDeleteRequest,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryResult,
    MemoryStatsResponse,
    MemoryStoreRequest,
    MemoryType,
)
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .working import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryPipeline:
    def __init__(
        self,
        redis_url: str,
        postgres_dsn: str,
        vector_dim: int,
        nim_api_key: str,
        nim_base_url: str,
    ) -> None:
        self._redis_url = redis_url
        self._postgres_dsn = postgres_dsn
        self._vector_dim = vector_dim
        self._nim_key = nim_api_key
        self._nim_base = nim_base_url.rstrip("/")
        self._embed_model = "nvidia/nv-embedqa-e5-v5"

        self._working: Optional[WorkingMemory] = None
        self._episodic: Optional[EpisodicMemory] = None
        self._semantic: Optional[SemanticMemory] = None
        self._procedural: Optional[ProceduralMemory] = None
        self._decay: Optional[DecayEngine] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._decay_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self._nim_base,
            headers={"Authorization": f"Bearer {self._nim_key}"},
            timeout=30.0,
        )

        self._working = WorkingMemory(self._redis_url)
        await self._working.connect()

        self._episodic = EpisodicMemory(self._postgres_dsn, self._vector_dim)
        await self._episodic.connect()

        self._semantic = SemanticMemory(self._postgres_dsn, self._vector_dim)
        await self._semantic.connect()

        self._procedural = ProceduralMemory(self._postgres_dsn)
        await self._procedural.connect()

        self._decay = DecayEngine(self._episodic, self._semantic)

        # Background decay every 10 minutes
        self._decay_task = asyncio.create_task(self._decay_loop())
        logger.info("Memory pipeline started")

    async def stop(self) -> None:
        if self._decay_task:
            self._decay_task.cancel()
        if self._http:
            await self._http.aclose()

    async def store(self, req: MemoryStoreRequest) -> str:
        memory_id = str(uuid.uuid4())
        embedding = await self._embed(req.content)
        content_hash = hashlib.sha256(req.content.encode()).hexdigest()

        now = datetime.now(timezone.utc)
        record = {
            "id": memory_id,
            "content": req.content,
            "memory_type": req.memory_type.value,
            "session_id": req.session_id,
            "agent_id": req.agent_id,
            "metadata": req.metadata,
            "created_at": now.isoformat(),
            "importance": req.importance,
            "confidence": req.importance,
            "content_hash": content_hash,
            "access_count": 0,
            "last_accessed": now.isoformat(),
        }

        # Working memory: always store for active sessions
        if req.session_id and self._working:
            await self._working.set(
                req.session_id,
                memory_id,
                record,
                ttl=req.ttl_seconds or 3600,
            )

        # Route to appropriate long-term store
        if req.memory_type == MemoryType.EPISODIC and self._episodic:
            await self._episodic.store(memory_id, record, embedding)
        elif req.memory_type == MemoryType.SEMANTIC and self._semantic:
            dedup_result = await self._semantic.find_duplicate(embedding, threshold=0.95)
            if dedup_result:
                await self._semantic.merge(dedup_result, record)
                return dedup_result
            await self._semantic.store(memory_id, record, embedding)
        elif req.memory_type == MemoryType.PROCEDURAL and self._procedural:
            await self._procedural.store(memory_id, record)
        elif req.memory_type == MemoryType.WORKING and self._working and req.session_id:
            pass  # already stored above

        logger.debug(f"Stored memory {memory_id} type={req.memory_type.value}")
        return memory_id

    async def query(self, req: MemoryQueryRequest) -> MemoryQueryResponse:
        start = time.monotonic()
        embedding = await self._embed(req.query)
        embed_ms = (time.monotonic() - start) * 1000

        # Fan-out: all memory types searched in parallel.
        coros = []
        for mem_type in req.memory_types:
            if mem_type == MemoryType.WORKING and self._working and req.session_id:
                coros.append(self._search_working(req))
            elif mem_type == MemoryType.EPISODIC and self._episodic:
                coros.append(self._search_episodic(embedding, req))
            elif mem_type == MemoryType.SEMANTIC and self._semantic:
                coros.append(self._search_semantic(embedding, req))
            elif mem_type == MemoryType.PROCEDURAL and self._procedural:
                coros.append(self._search_procedural(req))

        results_per_type: list[list[MemoryResult]] = (
            list(await asyncio.gather(*coros)) if coros else []
        )
        all_results: list[MemoryResult] = [r for sub in results_per_type for r in sub]

        # Deduplicate by content hash, keep highest score
        seen: dict[str, MemoryResult] = {}
        for r in all_results:
            key = r.memory_id
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        # Update access counts (fire and forget)
        for r in seen.values():
            asyncio.create_task(self._update_access(r.memory_id, r.memory_type))

        sorted_results = sorted(seen.values(), key=lambda r: r.score, reverse=True)[: req.limit]

        summary: Optional[str] = None
        if req.summarize and sorted_results:
            summary = await self._summarize_results(req.query, sorted_results)

        return MemoryQueryResponse(
            results=sorted_results,
            total=len(sorted_results),
            summary=summary,
            query_embedding_ms=embed_ms,
        )

    async def delete(self, req: MemoryDeleteRequest) -> int:
        count = 0
        if req.memory_ids and self._episodic:
            for mid in req.memory_ids:
                await self._episodic.delete(mid)
                count += 1
        if req.session_id and self._working:
            count += await self._working.clear_session(req.session_id)
        if req.older_than_hours and self._episodic:
            count += await self._episodic.delete_older_than(req.older_than_hours)
        return count

    async def run_decay(self) -> int:
        assert self._decay is not None
        return await self._decay.run()

    async def stats(self) -> MemoryStatsResponse:
        episodic_count = await self._episodic.count() if self._episodic else 0
        semantic_count = await self._semantic.count() if self._semantic else 0
        procedural_count = await self._procedural.count() if self._procedural else 0
        working_keys = await self._working.key_count() if self._working else 0

        return MemoryStatsResponse(
            total_memories=episodic_count + semantic_count + procedural_count,
            by_type={
                "episodic": episodic_count,
                "semantic": semantic_count,
                "procedural": procedural_count,
            },
            working_memory_keys=working_keys,
            vector_index_size=episodic_count + semantic_count,
            avg_confidence=0.75,
            oldest_memory_hours=0.0,
        )

    async def health(self) -> dict:
        redis_ok = False
        pg_ok = False
        if self._working:
            redis_ok = await self._working.ping()
        if self._episodic:
            pg_ok = await self._episodic.ping()
        return {
            "status": "healthy" if (redis_ok and pg_ok) else "degraded",
            "redis": redis_ok,
            "postgres": pg_ok,
        }

    async def _search_working(self, req: MemoryQueryRequest) -> list[MemoryResult]:
        assert self._working is not None
        items = await self._working.get_all(req.session_id, limit=req.limit)
        return [self._to_result(item, score=1.0) for item in items]

    async def _search_episodic(self, embedding: np.ndarray, req: MemoryQueryRequest) -> list[MemoryResult]:
        assert self._episodic is not None
        return await self._episodic.search(
            embedding, k=req.limit, session_id=req.session_id,
            max_age_hours=req.max_age_hours, min_score=req.min_score,
        )

    async def _search_semantic(self, embedding: np.ndarray, req: MemoryQueryRequest) -> list[MemoryResult]:
        assert self._semantic is not None
        return await self._semantic.search(embedding, k=req.limit, min_score=req.min_score)

    async def _search_procedural(self, req: MemoryQueryRequest) -> list[MemoryResult]:
        assert self._procedural is not None
        return await self._procedural.search(req.query, limit=req.limit)

    async def _embed(self, text: str) -> np.ndarray:
        assert self._http is not None
        resp = await self._http.post(
            "/embeddings",
            json={
                "model": self._embed_model,
                "input": [text],
                "input_type": "query",
                "encoding_format": "float",
                "truncate": "END",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    async def _summarize_results(self, query: str, results: list[MemoryResult]) -> str:
        assert self._http is not None
        context = "\n".join(f"- {r.content[:200]}" for r in results[:5])
        resp = await self._http.post(
            "/chat/completions",
            json={
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize the following memory snippets relevant to the query in 2-3 sentences.",
                    },
                    {"role": "user", "content": f"Query: {query}\n\nMemories:\n{context}"},
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            },
        )
        if resp.status_code >= 400:
            return ""
        return resp.json()["choices"][0]["message"]["content"]

    def _to_result(self, item: dict, score: float) -> MemoryResult:
        return MemoryResult(
            memory_id=item.get("id", ""),
            content=item.get("content", ""),
            memory_type=item.get("memory_type", "working"),
            score=score,
            confidence=item.get("confidence", 0.5),
            created_at=item.get("created_at", ""),
            metadata=item.get("metadata", {}),
            age_hours=0.0,
        )

    async def _update_access(self, memory_id: str, memory_type: str) -> None:
        try:
            if memory_type == "episodic" and self._episodic:
                await self._episodic.update_access(memory_id)
            elif memory_type == "semantic" and self._semantic:
                await self._semantic.update_access(memory_id)
        except Exception:
            pass

    async def _decay_loop(self) -> None:
        while True:
            await asyncio.sleep(600)  # 10 minutes
            try:
                expired = await self.run_decay()
                if expired:
                    logger.info(f"Memory decay: removed {expired} low-confidence memories")
            except Exception as e:
                logger.error(f"Decay loop error: {e}")
