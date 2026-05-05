"""3-tier model selector: NIM → OpenRouter Free → Ranked Fallback.

Implements the selection algorithm:

    def select_model(task_type):
        model = nim_models.get(task_type)
        if model and model.available():
            return model
        for m in openrouter_free_map[task_type]:
            if m.available():
                return m
        for ranked_model in finance_ranking:
            if ranked_model.supports(task_type):
                return ranked_model
        return "openrouter/auto"
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ..openrouter.models import CAPABILITY_MAP as OR_CAPABILITY_MAP
from .capability import Capability, TaskType
from .nim_models import NIM_CORE_MODELS, NIM_TASK_MAP, UnifiedModelSpec
from .ranking import RANKED_FALLBACKS, best_ranked_for_task

logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects the best available model for a task across all three tiers.

    Availability is determined by a lightweight ping cache so repeated
    selections within the same second don't re-probe the network.
    """

    def __init__(self, nim_available: bool = True, openrouter_available: bool = True) -> None:
        self._nim_ok = nim_available
        self._or_ok = openrouter_available
        # Simple in-memory unavailability blacklist: model_id → epoch_s when it failed
        self._blacklist: Dict[str, float] = {}
        self._blacklist_ttl = 60.0  # 60s cooldown before retry

    def mark_unavailable(self, model_id: str) -> None:
        import time
        self._blacklist[model_id] = time.time()
        logger.warning(f"Model '{model_id}' marked unavailable for {self._blacklist_ttl:.0f}s")

    def is_available(self, model_id: str) -> bool:
        import time
        ts = self._blacklist.get(model_id)
        if ts is None:
            return True
        if (time.time() - ts) >= self._blacklist_ttl:
            del self._blacklist[model_id]
            return True
        return False

    def select(
        self,
        task: TaskType,
        require_tools: bool = False,
        require_multimodal: bool = False,
        min_context: int = 0,
    ) -> Tuple[UnifiedModelSpec, str]:
        """Return (model_spec, provider_label) for the highest-priority available model."""

        # ── TIER 1: NIM ───────────────────────────────────────────────────────
        if self._nim_ok:
            nim_spec = NIM_TASK_MAP.get(task)
            if nim_spec and self.is_available(nim_spec.id):
                if not require_tools or nim_spec.supports_tools:
                    if nim_spec.max_context >= min_context:
                        logger.debug(f"[selector] tier=1 model={nim_spec.id} task={task.value}")
                        return nim_spec, "nim"
            # Try any NIM model that fits
            for spec in NIM_CORE_MODELS:
                if not self.is_available(spec.id):
                    continue
                if require_tools and not spec.supports_tools:
                    continue
                if spec.max_context < min_context:
                    continue
                if spec.supports_task(task):
                    logger.debug(f"[selector] tier=1 fallback model={spec.id}")
                    return spec, "nim"

        # ── TIER 2: OpenRouter free ────────────────────────────────────────────
        if self._or_ok:
            cap_key = task.value
            candidates = OR_CAPABILITY_MAP.get(cap_key, OR_CAPABILITY_MAP.get("reasoning", []))
            for or_model in candidates:
                if not self.is_available(or_model.id):
                    continue
                if require_tools and not or_model.supports_tools:
                    continue
                if or_model.context_length < min_context:
                    continue
                # Build a UnifiedModelSpec wrapper
                spec = UnifiedModelSpec(
                    id=or_model.id,
                    provider="openrouter",
                    tier=2,
                    capabilities=[Capability(c) for c in or_model.capabilities if c in [e.value for e in Capability]],
                    max_context=or_model.context_length,
                    supports_tools=or_model.supports_tools,
                    is_free=or_model.is_free,
                    description=or_model.description,
                )
                logger.debug(f"[selector] tier=2 model={spec.id} task={task.value}")
                return spec, "openrouter"

        # ── TIER 3: Ranked fallback ────────────────────────────────────────────
        ranked = best_ranked_for_task(task, require_tools=require_tools)
        logger.info(f"[selector] tier=3 fallback model={ranked.id} task={task.value}")
        return ranked, "ranked"

    def select_swarm(
        self,
        task: TaskType,
        n: int = 3,
    ) -> List[Tuple[UnifiedModelSpec, str]]:
        """Select N diverse models for parallel multi-model inference."""
        results: List[Tuple[UnifiedModelSpec, str]] = []
        seen_ids: set[str] = set()

        # Always try NIM first
        primary, provider = self.select(task)
        results.append((primary, provider))
        seen_ids.add(primary.id)

        # Fill with OR free models
        if self._or_ok:
            cap_key = task.value
            for or_model in OR_CAPABILITY_MAP.get(cap_key, []):
                if len(results) >= n:
                    break
                if or_model.id in seen_ids or not self.is_available(or_model.id):
                    continue
                spec = UnifiedModelSpec(
                    id=or_model.id,
                    provider="openrouter",
                    tier=2,
                    capabilities=[],
                    max_context=or_model.context_length,
                    is_free=True,
                )
                results.append((spec, "openrouter"))
                seen_ids.add(or_model.id)

        # Fill remaining with ranked fallbacks
        for ranked in RANKED_FALLBACKS:
            if len(results) >= n:
                break
            if ranked.id in seen_ids:
                continue
            if ranked.supports_task(task):
                results.append((ranked, "ranked"))
                seen_ids.add(ranked.id)

        return results[:n]


_selector: Optional[ModelSelector] = None


def get_selector() -> ModelSelector:
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector
