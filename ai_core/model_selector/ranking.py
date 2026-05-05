"""LEVEL 3 — Ranked fallback models (safety layer).

Ordered by: capability breadth, context length, tool support.
Used when both NIM and OpenRouter free tiers are unavailable.
"""
from __future__ import annotations

from typing import List

from .capability import Capability, TaskType, UnifiedModelSpec

RANKED_FALLBACKS: List[UnifiedModelSpec] = [
    UnifiedModelSpec(
        id="anthropic/claude-sonnet-4-5",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.REASONING, Capability.CODING, Capability.PLANNING,
            Capability.AUDIT, Capability.COMMUNICATION, Capability.TOOL_USE,
        ],
        max_context=200000,
        supports_tools=True,
        description="Claude Sonnet 4.5 — rank 1 failsafe",
    ),
    UnifiedModelSpec(
        id="mistralai/mistral-large",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.REASONING, Capability.CODING, Capability.TOOL_USE,
        ],
        max_context=131072,
        supports_tools=True,
        description="MiMo-V2-Pro equivalent — rank 2",
    ),
    UnifiedModelSpec(
        id="qwen/qwen2.5-plus",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.REASONING, Capability.CODING, Capability.TOOL_USE,
        ],
        max_context=131072,
        supports_tools=True,
        description="Qwen 3.6 Plus — rank 3",
    ),
    UnifiedModelSpec(
        id="minimax/minimax-m1",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.COMMUNICATION, Capability.REPORTING, Capability.REASONING,
        ],
        max_context=1000000,
        supports_tools=False,
        description="MiniMax M2.7 — rank 4, 1M context",
    ),
    UnifiedModelSpec(
        id="deepseek/deepseek-chat",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.REASONING, Capability.CODING, Capability.RESEARCH,
        ],
        max_context=131072,
        supports_tools=True,
        description="DeepSeek V3.2 — rank 5",
    ),
    UnifiedModelSpec(
        id="tencent/hunyuan-a13b-instruct",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.REASONING, Capability.RESEARCH,
        ],
        max_context=131072,
        supports_tools=False,
        description="Hy3 Preview — rank 6",
    ),
    UnifiedModelSpec(
        id="moonshot/kimi-k2",
        provider="ranked",
        tier=3,
        capabilities=[
            Capability.REASONING, Capability.CODING, Capability.TOOL_USE,
        ],
        max_context=131072,
        supports_tools=True,
        description="Kimi K2.6 — rank 7",
    ),
]


def best_ranked_for_task(task: TaskType, require_tools: bool = False) -> UnifiedModelSpec:
    """Return highest-ranked fallback that supports the task (and tools if required)."""
    for model in RANKED_FALLBACKS:
        if model.supports_task(task):
            if require_tools and not model.supports_tools:
                continue
            return model
    return RANKED_FALLBACKS[0]  # Claude as ultimate fallback
