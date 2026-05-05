"""Core LLM engine: wraps NIM API with circuit breaker, GPU allocation,
batching, streaming, and retry logic."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .circuit_breaker import CircuitBreaker
from .gpu_scheduler import GPUScheduler
from .model_router import ModelRouter, RoutingDecision
from .models import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .observability import LLM_LATENCY, TOKENS_USED

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        nim_api_key: str,
        router: ModelRouter,
        gpu_scheduler: GPUScheduler,
        nim_base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = nim_api_key
        self._base_url = nim_base_url.rstrip("/")
        self._timeout = timeout
        self._router = router
        self._gpu = gpu_scheduler
        self._cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(10)  # max concurrent requests

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        # Verify NIM is reachable
        try:
            resp = await self._client.get("/models")
            if resp.status_code >= 400:
                raise RuntimeError(f"NIM /models returned {resp.status_code}")
            logger.info(f"NIM engine started — base_url={self._base_url}")
        except Exception as e:
            logger.error(f"NIM unreachable at startup: {e}")
            raise

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
        self._gpu.stop()

    async def chat(self, req: ChatRequest) -> ChatResponse:
        decision = self._router.route(
            prompt=req.messages[-1].content if req.messages else "",
            tier_hint=req.tier,
            model_id=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        gpu_handle = self._gpu.allocate(decision.model_id, decision.gpu_mem_gb)
        start = time.monotonic()
        try:
            async with self._semaphore:
                result = await self._cb.call(
                    self._do_chat(decision, req)
                )
        finally:
            if gpu_handle:
                self._gpu.release(gpu_handle)

        latency_ms = (time.monotonic() - start) * 1000
        self._router.record_latency(decision.model_id, latency_ms, success=True)
        LLM_LATENCY.labels(model=decision.model_id, tier=decision.tier).observe(latency_ms / 1000)

        return ChatResponse(
            content=result["content"],
            model=decision.model_id,
            tier=decision.tier,
            usage=result.get("usage", {}),
            latency_ms=latency_ms,
            gpu_allocated=gpu_handle is not None,
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[str]:
        decision = self._router.route(
            prompt=req.messages[-1].content if req.messages else "",
            tier_hint=req.tier,
            model_id=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        gpu_handle = self._gpu.allocate(decision.model_id, decision.gpu_mem_gb)
        try:
            async for chunk in self._do_chat_stream(decision, req):
                yield chunk
        finally:
            if gpu_handle:
                self._gpu.release(gpu_handle)

    async def embed(self, req: EmbedRequest) -> EmbedResponse:
        embed_model = req.model or os.environ.get("MODEL_EMBED", "nvidia/nv-embedqa-e5-v5")
        payload = {
            "model": embed_model,
            "input": req.texts,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
        }

        async with self._semaphore:
            data = await self._post_with_retry("/embeddings", payload)

        embeddings = [item["embedding"] for item in data["data"]]
        dim = len(embeddings[0]) if embeddings else 0
        TOKENS_USED.labels(model=embed_model, type="embed").inc(len(req.texts))

        return EmbedResponse(embeddings=embeddings, model=embed_model, dim=dim)

    async def health(self) -> HealthResponse:
        nim_ok = False
        try:
            assert self._client is not None
            resp = await asyncio.wait_for(self._client.get("/models"), timeout=5.0)
            nim_ok = resp.status_code < 400
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if nim_ok else "degraded",
            nim_reachable=nim_ok,
            circuit_breaker=self._cb.state,
            active_gpus=len(self._gpu.status()["devices"]),
            queue_depth=self._semaphore._value,
        )

    def list_models(self) -> list[ModelInfoResponse]:
        return [
            ModelInfoResponse(
                id=m.id,
                tier=m.tier,
                gpu_mem_gb=m.gpu_mem_gb,
                avg_latency_ms=m.avg_latency_ms,
                cost_per_token=m.cost_per_token,
                available=m.available,
            )
            for m in self._router.list_models()
        ]

    async def _do_chat(self, decision: RoutingDecision, req: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": decision.model_id,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": decision.temperature,
            "max_tokens": decision.max_tokens,
            "top_p": decision.top_p,
            "stream": False,
        }
        if req.tools:
            payload["tools"] = req.tools

        data = await self._post_with_retry("/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        TOKENS_USED.labels(model=decision.model_id, type="chat").inc(
            usage.get("total_tokens", 0)
        )
        return {"content": content, "usage": usage}

    async def _do_chat_stream(
        self, decision: RoutingDecision, req: ChatRequest
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": decision.model_id,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": decision.temperature,
            "max_tokens": decision.max_tokens,
            "top_p": decision.top_p,
            "stream": True,
        }

        assert self._client is not None
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise RuntimeError(f"NIM stream error {resp.status_code}: {body.decode()}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    obj = json.loads(data_str)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield json.dumps({"delta": delta})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        assert self._client is not None
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TransportError, httpx.ReadTimeout)),
        ):
            with attempt:
                resp = await self._client.post(path, json=payload)
                if resp.status_code == 429:
                    await asyncio.sleep(2.0)
                    raise httpx.TransportError("rate-limited")
                if resp.status_code >= 400:
                    raise RuntimeError(f"NIM {path} {resp.status_code}: {resp.text[:500]}")
                return resp.json()
        raise RuntimeError("Retry loop exited without result")
