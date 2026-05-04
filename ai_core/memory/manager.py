"""High-level memory facade combining short and long term stores."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..logging_setup import logger
from ..nim_client import get_nim_client
from .long_term import LongTermMemory, MemoryRecord
from .short_term import ShortTermMemory


class MemoryManager:
    def __init__(self) -> None:
        s = get_settings()
        self.short = ShortTermMemory(max_messages=s.short_term_max)
        self.long = LongTermMemory(dim=s.long_term_dim, persist_dir=s.memory_dir)

    # ---- short-term passthrough ----

    def add_message(self, session_id: str, role: str, content: str, **meta: str) -> None:
        self.short.add(session_id, role, content, **meta)

    def history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        return self.short.messages(session_id, limit)

    # ---- long-term ----

    async def store(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        client = get_nim_client()
        emb = (await client.embed([text]))[0]
        rec = self.long.add(text=text, embedding=emb, tags=tags, meta=meta)
        logger.debug(f"stored memory id={rec.id} dim={len(emb)}")
        return rec

    async def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        client = get_nim_client()
        emb = (await client.embed([query]))[0]
        return self.long.search(emb, k=k, min_score=min_score)

    async def recall(self, query: str, k: int = 3) -> str:
        """Return a single concatenated context block for prompts."""
        results = await self.search(query, k=k)
        if not results:
            return ""
        chunks = [f"[mem score={r['score']:.2f}] {r['text']}" for r in results]
        return "\n".join(chunks)


_manager: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
