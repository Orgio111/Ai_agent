"""Advanced Memory Service: Redis working memory, PostgreSQL metadata,
FAISS vector search, with decay, deduplication, and confidence scoring."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .models import (
    MemoryDeleteRequest,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryStatsResponse,
    MemoryStoreRequest,
)
from .pipeline import MemoryPipeline

pipeline: MemoryPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global pipeline
    pipeline = MemoryPipeline(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        postgres_dsn=os.environ.get("DATABASE_URL", "postgresql://jarvis:jarvis@localhost:5432/jarvis"),
        vector_dim=int(os.environ.get("MEMORY_VECTOR_DIM", "1024")),
        nim_api_key=os.environ.get("NVIDIA_NIM_API_KEY", ""),
        nim_base_url=os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )
    await pipeline.start()
    yield
    await pipeline.stop()


app = FastAPI(title="JARVIS Memory Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/store")
async def store(req: MemoryStoreRequest) -> dict:
    assert pipeline is not None
    memory_id = await pipeline.store(req)
    return {"memory_id": memory_id, "status": "stored"}


@app.post("/query", response_model=MemoryQueryResponse)
async def query(req: MemoryQueryRequest) -> MemoryQueryResponse:
    assert pipeline is not None
    return await pipeline.query(req)


@app.delete("/delete")
async def delete(req: MemoryDeleteRequest) -> dict:
    assert pipeline is not None
    count = await pipeline.delete(req)
    return {"deleted": count}


@app.post("/decay")
async def run_decay() -> dict:
    assert pipeline is not None
    expired = await pipeline.run_decay()
    return {"decayed": expired}


@app.get("/stats", response_model=MemoryStatsResponse)
async def stats() -> MemoryStatsResponse:
    assert pipeline is not None
    return await pipeline.stats()


@app.get("/health")
async def health() -> dict:
    assert pipeline is not None
    return await pipeline.health()


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.environ.get("MEMORY_HOST", "0.0.0.0"),
        port=int(os.environ.get("MEMORY_PORT", "8003")),
        log_level="info",
    )
