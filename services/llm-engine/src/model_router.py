"""Model router: maps tier/task hints to NIM model IDs with latency tracking."""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import threading


@dataclass
class ModelSpec:
    id: str
    tier: str
    gpu_mem_gb: float
    cost_per_token: float
    avg_latency_ms: float = 1000.0
    success_rate: float = 1.0
    available: bool = True


_DEFAULT_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        id=os.environ.get("MODEL_COMPLEX", "meta/llama-3.1-70b-instruct"),
        tier="complex",
        gpu_mem_gb=40.0,
        cost_per_token=0.0008,
    ),
    ModelSpec(
        id=os.environ.get("MODEL_FAST", "mistralai/mistral-7b-instruct-v0.3"),
        tier="fast",
        gpu_mem_gb=14.0,
        cost_per_token=0.0001,
    ),
    ModelSpec(
        id=os.environ.get("MODEL_BALANCED", "mistralai/mixtral-8x7b-instruct-v0.1"),
        tier="balanced",
        gpu_mem_gb=24.0,
        cost_per_token=0.0004,
    ),
    ModelSpec(
        id=os.environ.get("MODEL_CODE", "meta/codellama-70b-instruct"),
        tier="code",
        gpu_mem_gb=40.0,
        cost_per_token=0.0006,
    ),
    ModelSpec(
        id=os.environ.get("MODEL_EMBED", "nvidia/nv-embedqa-e5-v5"),
        tier="embed",
        gpu_mem_gb=4.0,
        cost_per_token=0.00001,
    ),
]

_CODE_PATTERNS = re.compile(
    r"\b(code|function|class|def |import |algorithm|implement|debug|fix|refactor)\b",
    re.IGNORECASE,
)
_REASONING_PATTERNS = re.compile(
    r"\b(analyze|reason|explain|compare|evaluate|strategy|plan|research|complex)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    model_id: str
    tier: str
    gpu_mem_gb: float
    temperature: float
    max_tokens: int
    top_p: float
    reason: str


class ModelRouter:
    def __init__(self) -> None:
        self._registry: dict[str, ModelSpec] = {m.id: m for m in _DEFAULT_REGISTRY}
        self._tier_index: dict[str, list[ModelSpec]] = defaultdict(list)
        self._lock = threading.Lock()

        for spec in _DEFAULT_REGISTRY:
            self._tier_index[spec.tier].append(spec)

    def route(
        self,
        prompt: str = "",
        tier_hint: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RoutingDecision:
        with self._lock:
            # Explicit model override
            if model_id and model_id in self._registry:
                spec = self._registry[model_id]
                return self._make_decision(spec, temperature, max_tokens, "explicit")

            # Tier hint
            if tier_hint:
                candidates = self._tier_index.get(tier_hint, [])
                if candidates:
                    spec = self._best_available(candidates)
                    return self._make_decision(spec, temperature, max_tokens, f"tier_hint:{tier_hint}")

            # Heuristic routing
            inferred_tier = self._infer_tier(prompt)
            candidates = self._tier_index.get(inferred_tier, self._tier_index["balanced"])
            spec = self._best_available(candidates)
            return self._make_decision(spec, temperature, max_tokens, f"heuristic:{inferred_tier}")

    def _infer_tier(self, prompt: str) -> str:
        if len(prompt) < 100:
            return "fast"
        if _CODE_PATTERNS.search(prompt):
            return "code"
        if _REASONING_PATTERNS.search(prompt) or len(prompt) > 500:
            return "complex"
        return "balanced"

    def _best_available(self, candidates: list[ModelSpec]) -> ModelSpec:
        available = [c for c in candidates if c.available]
        if not available:
            available = candidates  # fallback even if marked unavailable
        # Score = success_rate / (avg_latency_ms + 1)
        return max(available, key=lambda m: m.success_rate / (m.avg_latency_ms + 1))

    def _make_decision(
        self,
        spec: ModelSpec,
        temperature: Optional[float],
        max_tokens: Optional[int],
        reason: str,
    ) -> RoutingDecision:
        tier_temps = {"fast": 0.3, "balanced": 0.5, "complex": 0.7, "code": 0.1}
        tier_tokens = {"fast": 1024, "balanced": 2048, "complex": 4096, "code": 4096}

        return RoutingDecision(
            model_id=spec.id,
            tier=spec.tier,
            gpu_mem_gb=spec.gpu_mem_gb,
            temperature=temperature if temperature is not None else tier_temps.get(spec.tier, 0.5),
            max_tokens=max_tokens if max_tokens is not None else tier_tokens.get(spec.tier, 2048),
            top_p=0.9,
            reason=reason,
        )

    def record_latency(self, model_id: str, latency_ms: float, success: bool) -> None:
        with self._lock:
            spec = self._registry.get(model_id)
            if spec is None:
                return
            # Exponential moving average
            alpha = 0.1
            spec.avg_latency_ms = alpha * latency_ms + (1 - alpha) * spec.avg_latency_ms
            if not success:
                spec.success_rate = max(0.0, spec.success_rate - 0.05)
            else:
                spec.success_rate = min(1.0, spec.success_rate + 0.01)

    def list_models(self) -> list[ModelSpec]:
        with self._lock:
            return list(self._registry.values())
