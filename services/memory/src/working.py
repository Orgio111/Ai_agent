"""Working memory layer: Redis-backed short-term session memory."""
from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis


class WorkingMemory:
    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(self._url, decode_responses=True)
        await self._redis.ping()

    def _key(self, session_id: str, memory_id: str) -> str:
        return f"jarvis:working:{session_id}:{memory_id}"

    def _session_index(self, session_id: str) -> str:
        return f"jarvis:working:idx:{session_id}"

    async def set(self, session_id: str, memory_id: str, record: dict, ttl: int = 3600) -> None:
        assert self._redis is not None
        key = self._key(session_id, memory_id)
        await self._redis.setex(key, ttl, json.dumps(record))
        await self._redis.sadd(self._session_index(session_id), memory_id)
        await self._redis.expire(self._session_index(session_id), ttl)

    async def get(self, session_id: str, memory_id: str) -> Optional[dict]:
        assert self._redis is not None
        raw = await self._redis.get(self._key(session_id, memory_id))
        return json.loads(raw) if raw else None

    async def get_all(self, session_id: str, limit: int = 50) -> list[dict]:
        assert self._redis is not None
        ids = await self._redis.smembers(self._session_index(session_id))
        results = []
        for mid in list(ids)[:limit]:
            raw = await self._redis.get(self._key(session_id, mid))
            if raw:
                results.append(json.loads(raw))
        return results

    async def clear_session(self, session_id: str) -> int:
        assert self._redis is not None
        ids = await self._redis.smembers(self._session_index(session_id))
        count = 0
        for mid in ids:
            await self._redis.delete(self._key(session_id, mid))
            count += 1
        await self._redis.delete(self._session_index(session_id))
        return count

    async def key_count(self) -> int:
        assert self._redis is not None
        return await self._redis.dbsize()

    async def ping(self) -> bool:
        try:
            assert self._redis is not None
            await self._redis.ping()
            return True
        except Exception:
            return False
