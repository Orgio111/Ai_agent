"""FastAPI entrypoint for the AI core.

Run from the repo root:
    uvicorn ai_core.main:app --host 0.0.0.0 --port 8000 --reload
or:
    python -m ai_core
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_setup import logger, setup_logging
from .memory import get_memory
from .nim_client import get_nim_client
from .orchestrator import get_orchestrator
from .tools import get_registry

# ---------------- Schemas ----------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    tier: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


class AgentRequest(BaseModel):
    goal: str = Field(..., description="High-level goal for the agent loop")
    session_id: str = "default"


class MemoryStoreRequest(BaseModel):
    text: str
    tags: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    query: str
    k: int = 5
    min_score: float = 0.0


class ToolRunRequest(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


# ---------------- Lifespan ----------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    s = get_settings()
    # Warm singletons.
    get_memory()
    get_registry()
    get_orchestrator()
    logger.info(f"AI core starting on {s.ai_core_host}:{s.ai_core_port}")
    yield
    client = get_nim_client()
    await client.close()
    logger.info("AI core shut down cleanly")


# ---------------- App ----------------


app = FastAPI(
    title="Hybrid AI System - Core",
    description="Multi-agent autonomous AI core powered by NVIDIA NIM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    s = get_settings()
    return {
        "service": "hybrid-ai-core",
        "version": "1.0.0",
        "models": {k: v.model for k, v in s.models.routing.items()},
        "endpoints": [
            "/health", "/chat", "/chat/stream", "/agent/run",
            "/memory/store", "/memory/search", "/tools", "/tools/run",
        ],
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "nim_configured": bool(s.nim_api_key),
        "memory_size": len(get_memory().long),
    }


# ---------------- Chat ----------------


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    if req.stream:
        raise HTTPException(400, "use /chat/stream for streaming")
    client = get_nim_client()
    try:
        result = await client.chat(
            [m.model_dump() for m in req.messages],
            tier=req.tier,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(502, f"NIM error: {e}") from e
    return result


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    client = get_nim_client()

    async def gen():
        try:
            async for chunk in client.chat_stream(
                [m.model_dump() for m in req.messages],
                tier=req.tier,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                yield chunk
        except Exception as e:
            yield f"\n[stream error: {e}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


# ---------------- Agent loop ----------------


@app.post("/agent/run")
async def agent_run(req: AgentRequest) -> Dict[str, Any]:
    orch = get_orchestrator()
    try:
        result = await orch.run(req.goal, session_id=req.session_id)
    except Exception as e:
        logger.exception("agent run failed")
        raise HTTPException(500, f"agent error: {e}") from e
    return result.to_dict()


# ---------------- Memory ----------------


@app.post("/memory/store")
async def memory_store(req: MemoryStoreRequest) -> Dict[str, Any]:
    mem = get_memory()
    try:
        rec = await mem.store(req.text, tags=req.tags, meta=req.meta)
    except Exception as e:
        raise HTTPException(502, f"embedding/store failed: {e}") from e
    return {"id": rec.id, "ok": True}


@app.post("/memory/search")
async def memory_search(req: MemorySearchRequest) -> Dict[str, Any]:
    mem = get_memory()
    try:
        hits = await mem.search(req.query, k=req.k, min_score=req.min_score)
    except Exception as e:
        raise HTTPException(502, f"search failed: {e}") from e
    return {"hits": hits, "count": len(hits)}


# ---------------- Tools ----------------


@app.get("/tools")
async def tools_list() -> Dict[str, Any]:
    return {"tools": get_registry().list()}


@app.post("/tools/run")
async def tools_run(req: ToolRunRequest) -> Dict[str, Any]:
    res = await get_registry().run(req.name, **req.args)
    return res.to_dict()


# ---------------- CLI entry ----------------


def main() -> None:
    s = get_settings()
    uvicorn.run(
        "ai_core.main:app",
        host=s.ai_core_host,
        port=s.ai_core_port,
        reload=False,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    main()
