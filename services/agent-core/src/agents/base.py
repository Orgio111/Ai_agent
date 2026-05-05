"""Base agent with NIM integration, memory retrieval, and tool dispatch."""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Optional

import httpx


class AgentBase:
    name: str = "base"
    tier: str = "balanced"
    system_prompt: str = "You are a helpful AI agent."

    def __init__(
        self,
        nim_client: httpx.AsyncClient,
        memory_url: str,
        tool_url: str,
        nim_base_url: str,
    ) -> None:
        self._nim = nim_client
        self._memory_url = memory_url
        self._tool_url = tool_url
        self._nim_base = nim_base_url
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._latencies: list[float] = []

    @property
    def avg_latency_ms(self) -> float:
        return sum(self._latencies[-50:]) / max(len(self._latencies[-50:]), 1)

    async def run(
        self,
        prompt: str,
        session_id: str,
        history: list[dict],
        context: dict[str, Any],
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        memory_ctx = await self._retrieve_memory(prompt, session_id)
        messages = self._build_messages(prompt, history, memory_ctx)

        payload: dict[str, Any] = {
            "messages": messages,
            "tier": self.tier,
            "session_id": session_id,
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = await self._nim.post("/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["content"]
            usage = data.get("usage", {})
        except Exception as e:
            self.tasks_failed += 1
            raise RuntimeError(f"[{self.name}] NIM call failed: {e}") from e

        latency = (time.monotonic() - start) * 1000
        self._latencies.append(latency)
        self.tasks_completed += 1

        # Store result in episodic memory
        asyncio.create_task(
            self._store_memory(
                f"Agent {self.name} response: {content[:300]}",
                session_id,
                memory_type="episodic",
            )
        )

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency,
            "tool_calls": self._extract_tool_calls(content),
        }

    async def run_stream(
        self,
        prompt: str,
        session_id: str,
        history: list[dict],
        context: dict[str, Any],
    ) -> AsyncIterator[str]:
        memory_ctx = await self._retrieve_memory(prompt, session_id)
        messages = self._build_messages(prompt, history, memory_ctx)

        payload = {
            "messages": messages,
            "tier": self.tier,
            "session_id": session_id,
            "stream": True,
        }

        async with self._nim.stream("POST", "/chat/stream", json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data and data != "[DONE]":
                        yield data

    def _build_messages(
        self,
        prompt: str,
        history: list[dict],
        memory_ctx: str,
    ) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if memory_ctx:
            msgs.append({"role": "system", "content": f"Relevant context:\n{memory_ctx}"})
        msgs.extend(history[-20:])  # last 20 turns
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def _retrieve_memory(self, query: str, session_id: str) -> str:
        try:
            async with httpx.AsyncClient(base_url=self._memory_url, timeout=5.0) as client:
                resp = await client.post(
                    "/query",
                    json={
                        "query": query,
                        "session_id": session_id,
                        "limit": 5,
                        "min_score": 0.6,
                    },
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    return "\n".join(r["content"][:200] for r in results[:3])
        except Exception:
            pass
        return ""

    async def _store_memory(
        self, content: str, session_id: str, memory_type: str = "episodic"
    ) -> None:
        try:
            async with httpx.AsyncClient(base_url=self._memory_url, timeout=3.0) as client:
                await client.post(
                    "/store",
                    json={
                        "content": content,
                        "memory_type": memory_type,
                        "session_id": session_id,
                        "importance": 0.5,
                    },
                )
        except Exception:
            pass

    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        try:
            async with httpx.AsyncClient(base_url=self._tool_url, timeout=30.0) as client:
                resp = await client.post(
                    "/execute",
                    json={"tool_name": tool_name, "args": args},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e), "tool_name": tool_name}

    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict]:
        pattern = re.compile(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL | re.IGNORECASE
        )
        calls = []
        for match in pattern.finditer(text):
            try:
                calls.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

        if not calls:
            fence = re.search(r"```(?:json)?\s*(\{.*?\"tool\".*?\})\s*```", text, re.DOTALL)
            if fence:
                try:
                    calls.append(json.loads(fence.group(1)))
                except json.JSONDecodeError:
                    pass

        return calls
