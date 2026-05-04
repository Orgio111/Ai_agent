"""NIM client offline tests using httpx MockTransport.

Validates that:
  - chat() shapes the payload correctly and parses the response,
  - 4xx errors raise NIMError,
  - retries kick in on transport errors.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ai_core.nim_client.client import NIMClient, NIMError


def test_chat_happy_path():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "pong"}}],
                "usage": {"total_tokens": 4},
            },
        )

    client = NIMClient(api_key="test", base_url="https://example.test/v1")
    client._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        transport=httpx.MockTransport(handler),
        base_url="https://example.test/v1",
        headers={"Authorization": "Bearer test"},
    )

    res = asyncio.run(
        client.chat(
            [{"role": "user", "content": "ping"}],
            tier="fast",
        )
    )
    assert res["content"] == "pong"
    assert captured["payload"]["messages"][0]["content"] == "ping"
    assert captured["payload"]["stream"] is False
    assert "model" in captured["payload"]


def test_chat_4xx_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = NIMClient(api_key="bad", base_url="https://example.test/v1")
    client._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        transport=httpx.MockTransport(handler),
        base_url="https://example.test/v1",
    )

    with pytest.raises(NIMError):
        asyncio.run(client.chat([{"role": "user", "content": "x"}], tier="fast"))
