"""OpenRouter API client — OpenAI-compatible endpoint.

Supports free models (primary swarm layer) and paid ranked fallbacks.
Uses connection pooling and exponential-backoff retry identical to NIMClient.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..logging_setup import logger

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_APP_TITLE = "JARVIS-AI"
_APP_REFERER = "https://github.com/Orgio111/Ai_agent"


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    """Async OpenRouter client with persistent connection pool and retry."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 120.0,
        base_url: str = OPENROUTER_BASE,
    ) -> None:
        self.api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": _APP_REFERER,
                    "X-Title": _APP_TITLE,
                },
                timeout=self._timeout,
                limits=httpx.Limits(
                    max_connections=200,
                    max_keepalive_connections=100,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        tools: Optional[List[Dict[str, Any]]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Non-streaming chat completion."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        payload.update(extra)

        data = await self._post_with_retry("/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise OpenRouterError(f"Malformed response: {data}") from e

        return {
            "content": content,
            "model": model,
            "usage": data.get("usage", {}),
            "raw": data,
        }

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        **extra: Any,
    ) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        payload.update(extra)

        client = await self._ensure()
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise OpenRouterError(f"Stream error {resp.status_code}: {body.decode(errors='ignore')}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

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
                    raise httpx.TransportError("rate-limited")
                if resp.status_code >= 400:
                    raise OpenRouterError(f"OpenRouter {path} {resp.status_code}: {resp.text[:300]}")
                try:
                    return resp.json()
                except json.JSONDecodeError as e:
                    raise OpenRouterError(f"Non-JSON response: {resp.text[:200]}") from e
        raise OpenRouterError("Retry loop exited without result")


_singleton: Optional[OpenRouterClient] = None


def get_openrouter_client(api_key: str = "") -> OpenRouterClient:
    global _singleton
    if _singleton is None:
        import os
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        _singleton = OpenRouterClient(api_key=key)
        logger.info("OpenRouter client initialised")
    return _singleton
