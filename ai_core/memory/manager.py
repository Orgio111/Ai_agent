"""High-level memory facade combining short and long term stores."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..logging_setup import logger
from ..nim_client import get_nim_client
from .long_term import LongTermMemory, MemoryRecord
from .short_term import SessionMemory


class MemoryManager:
    def __init__(self, dim: Optional[int] = None) -> None:
        s = get_settings()
        self._session_mem = SessionMemory(max_messages=s.short_term_max)
        effective_dim = dim if dim is not None else s.long_term_dim
        self.long = LongTermMemory(dim=effective_dim)

    # ---- short-term passthrough ----

    def add_message(self, session_id: str, role: str, content: str, **meta: str) -> None:
        self._session_mem.add(session_id, role, content, **meta)

    def history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        return self._session_mem.history(session_id, limit)

    # ---- long-term (test-compatible API) ----

    async def store(
        self,
        content: Optional[str] = None,
        session_id: str = "default",
        memory_type: str = "episodic",
        text: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        actual_text = content or text or ""
        client = get_nim_client()
        emb = (await client.embed([actual_text]))[0]
        effective_tags = tags or [session_id, memory_type]
        rec = await asyncio.to_thread(
            self.long.add, text=actual_text, embedding=emb, tags=effective_tags, meta=meta or {}
        )
        logger.debug(f"stored memory id={rec.id} type={memory_type}")
        return rec

    async def query(
        self,
        query: str,
        session_id: str = "default",
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        client = get_nim_client()
        emb = (await client.embed([query]))[0]
        return self.long.search(emb, k=limit, min_score=min_score)

    async def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        return await self.query(query=query, limit=k, min_score=min_score)

    async def recall(self, query: str, k: int = 3) -> str:
        results = await self.query(query=query, limit=k)
        if not results:
            return ""
        return "\n".join(f"[mem score={r['score']:.2f}] {r['text']}" for r in results)


_manager: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
