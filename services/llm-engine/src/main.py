"""LLM Engine: FastAPI service wrapping NVIDIA NIM with GPU scheduling,
circuit breakers, streaming, and gRPC support."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .circuit_breaker import CircuitBreakerOpenError
from .engine import LLMEngine
from .gpu_scheduler import GPUScheduler
from .model_router import ModelRouter
from .models import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .observability import setup_telemetry, track_request

engine: LLMEngine | None = None
gpu_scheduler: GPUScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global engine, gpu_scheduler

    nim_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    if not nim_key or nim_key.startswith("nvapi-your"):
        raise RuntimeError(
            "NVIDIA_NIM_API_KEY is not set or is a placeholder. "
            "Set a valid NVIDIA NIM API key to start the LLM engine."
        )

    setup_telemetry()
    gpu_scheduler = GPUScheduler()
    router = ModelRouter()
    engine = LLMEngine(nim_api_key=nim_key, router=router, gpu_scheduler=gpu_scheduler)
    await engine.start()

    yield

    await engine.stop()


app = FastAPI(title="JARVIS LLM Engine", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    assert engine is not None
    start = time.monotonic()
    try:
        result = await engine.chat(req)
        latency = time.monotonic() - start
        track_request("chat", req.model or "auto", latency, success=True)
        return result
    except CircuitBreakerOpenError as e:
        track_request("chat", req.model or "auto", time.monotonic() - start, success=False)
        raise HTTPException(503, detail=f"Circuit breaker open: {e}")
    except Exception as e:
        track_request("chat", req.model or "auto", time.monotonic() - start, success=False)
        raise HTTPException(500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    assert engine is not None

    async def generate() -> AsyncIterator[str]:
        try:
            async for chunk in engine.chat_stream(req):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    assert engine is not None
    start = time.monotonic()
    try:
        result = await engine.embed(req)
        track_request("embed", req.model or "nv-embed", time.monotonic() - start, success=True)
        return result
    except Exception as e:
        track_request("embed", req.model or "nv-embed", time.monotonic() - start, success=False)
        raise HTTPException(500, detail=str(e))


@app.get("/models", response_model=list[ModelInfoResponse])
async def list_models() -> list[ModelInfoResponse]:
    assert engine is not None
    return engine.list_models()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    assert engine is not None
    return await engine.health()


@app.get("/gpu/status")
async def gpu_status():
    assert gpu_scheduler is not None
    return gpu_scheduler.status()


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.environ.get("LLM_ENGINE_HOST", "0.0.0.0"),
        port=int(os.environ.get("LLM_ENGINE_PORT", "8002")),
        reload=False,
        log_level="info",
    )
