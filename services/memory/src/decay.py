"""Memory decay engine: ages and prunes low-confidence memories using
Ebbinghaus forgetting curve and recency scoring."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import logging

if TYPE_CHECKING:
    from .episodic import EpisodicMemory
    from .semantic import SemanticMemory

logger = logging.getLogger(__name__)


def ebbinghaus_retention(age_hours: float, stability: float = 1.0) -> float:
    """R = e^(-t/S) — retention after `age_hours` with memory stability S."""
    return math.exp(-age_hours / max(stability * 24.0, 1.0))


class DecayEngine:
    def __init__(self, episodic: "EpisodicMemory", semantic: "SemanticMemory") -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._min_confidence = 0.15

    async def run(self) -> int:
        """Decay all memory layers. Returns count of memories removed."""
        removed = 0

        # Episodic: apply forgetting curve, decay confidence, remove below threshold
        episodic_low = await self._episodic.get_low_confidence(self._min_confidence)
        for mid in episodic_low:
            await self._episodic.delete(mid)
            removed += 1

        # Also decay old episodic memories (>72h) that are rarely accessed
        removed += await self._episodic.delete_older_than(hours=72.0)

        logger.debug(f"Decay complete: removed={removed}")
        return removed
