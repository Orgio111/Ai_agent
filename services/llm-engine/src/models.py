from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    tier: Optional[str] = None  # "fast" | "balanced" | "complex" | "code"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    session_id: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None

    @field_validator("temperature")
    @classmethod
    def clamp_temp(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return max(0.0, min(2.0, v))
        return v


class ChatResponse(BaseModel):
    content: str
    model: str
    tier: str
    usage: dict[str, Any] = {}
    latency_ms: float = 0.0
    gpu_allocated: bool = False


class EmbedRequest(BaseModel):
    texts: list[str]
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


class ModelInfoResponse(BaseModel):
    id: str
    tier: str
    gpu_mem_gb: float
    avg_latency_ms: float
    cost_per_token: float
    available: bool


class HealthResponse(BaseModel):
    status: str
    nim_reachable: bool
    circuit_breaker: str
    active_gpus: int
    queue_depth: int
