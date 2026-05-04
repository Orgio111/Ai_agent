from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = {}
    ttl_seconds: Optional[int] = None
    importance: float = 0.5


class MemoryResult(BaseModel):
    memory_id: str
    content: str
    memory_type: str
    score: float
    confidence: float
    created_at: str
    metadata: dict[str, Any] = {}
    age_hours: float = 0.0


class MemoryQueryRequest(BaseModel):
    query: str
    memory_types: list[MemoryType] = [MemoryType.EPISODIC, MemoryType.SEMANTIC]
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    limit: int = 10
    min_score: float = 0.5
    max_age_hours: Optional[float] = None
    summarize: bool = False


class MemoryQueryResponse(BaseModel):
    results: list[MemoryResult]
    total: int
    summary: Optional[str] = None
    query_embedding_ms: float = 0.0


class MemoryDeleteRequest(BaseModel):
    memory_ids: Optional[list[str]] = None
    session_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    older_than_hours: Optional[float] = None


class MemoryStatsResponse(BaseModel):
    total_memories: int
    by_type: dict[str, int]
    working_memory_keys: int
    vector_index_size: int
    avg_confidence: float
    oldest_memory_hours: float
