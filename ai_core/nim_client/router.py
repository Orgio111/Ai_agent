"""Heuristic + explicit model router for NVIDIA NIM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import ModelTier, get_settings


@dataclass
class RoutingDecision:
    tier: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    reason: str


class ModelRouter:
    """Selects a tier based on explicit hint or heuristic keyword scan."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _tier(self, name: str) -> ModelTier:
        routing = self.settings.models.routing
        if name not in routing:
            # Fallback chain.
            for fallback in ("balanced", "fast", "complex"):
                if fallback in routing:
                    return routing[fallback]
            raise RuntimeError("No models configured in routing")
        return routing[name]

    def route(self, prompt: str, tier_hint: Optional[str] = None) -> RoutingDecision:
        if tier_hint:
            tier = self._tier(tier_hint)
            return RoutingDecision(
                tier=tier_hint,
                model=tier.model,
                temperature=tier.temperature,
                max_tokens=tier.max_tokens,
                top_p=tier.top_p,
                reason=f"explicit hint:{tier_hint}",
            )

        h = self.settings.models.heuristics
        text = prompt.lower()

        if any(kw in text for kw in h.code_keywords):
            tier = self._tier("code")
            return RoutingDecision("code", tier.model, tier.temperature, tier.max_tokens, tier.top_p, "code-keyword")

        if any(kw in text for kw in h.complex_keywords) or len(prompt) > 1200:
            tier = self._tier("complex")
            return RoutingDecision("complex", tier.model, tier.temperature, tier.max_tokens, tier.top_p, "complex-keyword/long")

        if any(kw in text for kw in h.fast_keywords) and len(prompt) < 200:
            tier = self._tier("fast")
            return RoutingDecision("fast", tier.model, tier.temperature, tier.max_tokens, tier.top_p, "fast-keyword/short")

        tier = self._tier("balanced")
        return RoutingDecision("balanced", tier.model, tier.temperature, tier.max_tokens, tier.top_p, "default-balanced")


_router: Optional[ModelRouter] = None


def route_for(prompt: str, tier_hint: Optional[str] = None) -> RoutingDecision:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router.route(prompt, tier_hint)
