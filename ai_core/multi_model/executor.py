"""Multi-model parallel executor.

Fans out a single prompt to N models simultaneously, then aggregates.
Supports NIM and OpenRouter as backends.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_setup import logger
from ..model_selector.capability import TaskType, UnifiedModelSpec
from ..model_selector.selector import ModelSelector, get_selector
from ..nim_client import get_nim_client
from ..openrouter.client import get_openrouter_client


@dataclass
class ModelResult:
    model_id: str
    provider: str
    content: str
    latency_ms: float
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content)


@dataclass
class AggregatedResult:
    task: str
    results: List[ModelResult]
    consensus: str
    best: ModelResult
    latency_ms: float

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.ok)


class MultiModelExecutor:
    """Executes the same prompt across multiple models concurrently."""

    def __init__(
        self,
        selector: Optional[ModelSelector] = None,
        timeout_per_model: float = 60.0,
    ) -> None:
        self._selector = selector or get_selector()
        self._timeout = timeout_per_model

    async def run(
        self,
        messages: List[Dict[str, str]],
        task: TaskType,
        n_models: int = 3,
        temperature: float = 0.5,
        max_tokens: int = 2048,
    ) -> AggregatedResult:
        """Fan out to n_models in parallel and aggregate responses."""
        candidates = self._selector.select_swarm(task, n=n_models)
        start = time.monotonic()

        tasks = [
            self._call_model(messages, spec, provider, temperature, max_tokens)
            for spec, provider in candidates
        ]
        raw_results: List[ModelResult] = list(await asyncio.gather(*tasks))

        elapsed = (time.monotonic() - start) * 1000
        good = [r for r in raw_results if r.ok]

        if not good:
            logger.error(f"[multi_model] all {len(raw_results)} models failed for {task.value}")
            best = raw_results[0]
        else:
            # Pick longest non-empty response as "best" (heuristic for completeness)
            best = max(good, key=lambda r: len(r.content))

        consensus = self._aggregate(good or raw_results)
        return AggregatedResult(
            task=task.value,
            results=raw_results,
            consensus=consensus,
            best=best,
            latency_ms=elapsed,
        )

    async def _call_model(
        self,
        messages: List[Dict[str, str]],
        spec: UnifiedModelSpec,
        provider: str,
        temperature: float,
        max_tokens: int,
    ) -> ModelResult:
        start = time.monotonic()
        try:
            if provider == "nim":
                client = get_nim_client()
                resp = await asyncio.wait_for(
                    client.chat(messages, model=spec.id, temperature=temperature, max_tokens=max_tokens),
                    timeout=self._timeout,
                )
                return ModelResult(
                    model_id=spec.id,
                    provider="nim",
                    content=resp.get("content", ""),
                    usage=resp.get("usage", {}),
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            else:
                # openrouter or ranked (both use OpenRouter endpoint)
                or_client = get_openrouter_client()
                resp = await asyncio.wait_for(
                    or_client.chat(messages, model=spec.id, temperature=temperature, max_tokens=max_tokens),
                    timeout=self._timeout,
                )
                return ModelResult(
                    model_id=spec.id,
                    provider=provider,
                    content=resp.get("content", ""),
                    usage=resp.get("usage", {}),
                    latency_ms=(time.monotonic() - start) * 1000,
                )
        except Exception as e:
            logger.warning(f"[multi_model] {spec.id} failed: {e}")
            self._selector.mark_unavailable(spec.id)
            return ModelResult(
                model_id=spec.id,
                provider=provider,
                content="",
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(e),
            )

    @staticmethod
    def _aggregate(results: List[ModelResult]) -> str:
        """Simple aggregation: if all agree on a short answer, return it;
        otherwise return the longest (most detailed) response."""
        if not results:
            return ""
        if len(results) == 1:
            return results[0].content

        contents = [r.content.strip() for r in results if r.content.strip()]
        if not contents:
            return ""

        # Exact consensus
        if len(set(contents)) == 1:
            return contents[0]

        # Return the response with the highest word count (most complete)
        return max(contents, key=lambda c: len(c.split()))


_executor: Optional[MultiModelExecutor] = None


def get_executor() -> MultiModelExecutor:
    global _executor
    if _executor is None:
        _executor = MultiModelExecutor()
    return _executor
