"""Async NVIDIA NIM client with retries, streaming, and embeddings.

Uses the OpenAI-compatible NIM endpoint (`/v1/chat/completions`,
`/v1/embeddings`) via httpx so we control timeouts/retries explicitly.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..logging_setup import logger
from .router import RoutingDecision, route_for


class NIMError(RuntimeError):
    pass


class NIMClient:
    """Thin async client for NVIDIA NIM, OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        s = get_settings()
        self.api_key = api_key or s.nim_api_key
        self.base_url = (base_url or s.nim_base_url).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ---------------- Chat ----------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tier: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Single (non-stream) chat completion. Returns parsed JSON dict
        with `content`, `model`, `tier`, `usage`."""
        if stream:
            chunks: List[str] = []
            async for delta in self.chat_stream(
                messages, tier=tier, model=model,
                temperature=temperature, max_tokens=max_tokens, top_p=top_p, **extra
            ):
                chunks.append(delta)
            return {"content": "".join(chunks), "model": model or "", "tier": tier or "", "usage": {}}

        decision = self._decision(messages, tier, model, temperature, max_tokens, top_p)
        payload = {
            "model": decision.model,
            "messages": messages,
            "temperature": decision.temperature,
            "max_tokens": decision.max_tokens,
            "top_p": decision.top_p,
            "stream": False,
        }
        payload.update(extra)

        data = await self._post_with_retry("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise NIMError(f"Malformed NIM response: {data}") from e

        return {
            "content": content,
            "model": decision.model,
            "tier": decision.tier,
            "usage": data.get("usage", {}),
            "raw": data,
        }

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tier: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **extra: Any,
    ) -> AsyncIterator[str]:
        decision = self._decision(messages, tier, model, temperature, max_tokens, top_p)
        payload = {
            "model": decision.model,
            "messages": messages,
            "temperature": decision.temperature,
            "max_tokens": decision.max_tokens,
            "top_p": decision.top_p,
            "stream": True,
        }
        payload.update(extra)

        client = await self._ensure()
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise NIMError(f"NIM stream error {resp.status_code}: {body.decode(errors='ignore')}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = obj["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError):
                    delta = ""
                if delta:
                    yield delta

    # ---------------- Embeddings ----------------

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        s = get_settings()
        emb_model = model or s.models.embedding.model
        payload = {
            "model": emb_model,
            "input": texts,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
        }
        data = await self._post_with_retry("/embeddings", payload)
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as e:
            raise NIMError(f"Malformed embedding response: {data}") from e

    # ---------------- Internals ----------------

    def _decision(
        self,
        messages: List[Dict[str, str]],
        tier: Optional[str],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
    ) -> RoutingDecision:
        if model:
            # Explicit model overrides the router; still produce a decision.
            return RoutingDecision(
                tier=tier or "explicit",
                model=model,
                temperature=temperature if temperature is not None else 0.5,
                max_tokens=max_tokens if max_tokens is not None else 2048,
                top_p=top_p if top_p is not None else 0.9,
                reason="explicit model",
            )
        # Use the last user message for routing heuristics.
        prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break
        decision = route_for(prompt, tier_hint=tier)
        if temperature is not None:
            decision.temperature = temperature
        if max_tokens is not None:
            decision.max_tokens = max_tokens
        if top_p is not None:
            decision.top_p = top_p
        return decision

    async def _post_with_retry(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._ensure()
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TransportError, httpx.ReadTimeout)),
        ):
            with attempt:
                resp = await client.post(path, json=payload)
                if resp.status_code == 429:
                    # Rate limit: brief backoff, then let tenacity retry.
                    await asyncio.sleep(1.5)
                    raise httpx.TransportError("rate-limited")
                if resp.status_code >= 400:
                    raise NIMError(f"NIM {path} {resp.status_code}: {resp.text}")
                try:
                    return resp.json()
                except json.JSONDecodeError as e:
                    raise NIMError(f"NIM non-JSON response: {resp.text}") from e
        raise NIMError("Unreachable: retry loop exited without result")


_singleton: Optional[NIMClient] = None


def get_nim_client() -> NIMClient:
    global _singleton
    if _singleton is None:
        _singleton = NIMClient()
        logger.info(f"NIM client initialized → {_singleton.base_url}")
    return _singleton
