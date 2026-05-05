"""OpenRouter free + paid model catalog.

Models are grouped by task capability so the selector can pick the best
available model for a given workload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ORModel:
    id: str
    is_free: bool
    capabilities: List[str] = field(default_factory=list)
    context_length: int = 32768
    supports_tools: bool = False
    description: str = ""


# ─── FREE tier (LEVEL 2 swarm) ────────────────────────────────────────────────

REASONING_FREE: List[ORModel] = [
    ORModel("tencent/hunyuan-a13b-instruct:free",    is_free=True, capabilities=["reasoning","research"],       context_length=131072, description="Hy3 Preview — deep reasoning"),
    ORModel("qwen/qwen3-235b-a22b:free",              is_free=True, capabilities=["reasoning","research","coding"], context_length=131072, description="Qwen3 235B MoE — strong reasoning"),
    ORModel("liquid/lfm-thinking:free",               is_free=True, capabilities=["reasoning"],                 context_length=32768,  description="LFM Thinking chain-of-thought"),
]

CODING_FREE: List[ORModel] = [
    ORModel("qwen/qwen2.5-coder-32b-instruct:free",  is_free=True, capabilities=["coding","engineering"],       context_length=131072, description="Qwen Coder 32B"),
    ORModel("poolside/laguna-m.1:free",               is_free=True, capabilities=["coding","engineering"],       context_length=65536,  description="Poolside Laguna engineering model"),
]

MULTIMODAL_FREE: List[ORModel] = [
    ORModel("google/gemma-3-27b-it:free",             is_free=True, capabilities=["multimodal","perception"],   context_length=131072, description="Gemma 3 27B vision+text"),
    ORModel("google/gemma-3-12b-it:free",             is_free=True, capabilities=["multimodal","lightweight"],  context_length=131072, description="Gemma 3 12B"),
    ORModel("baidu/qianfan-vl-plus:free",             is_free=True, capabilities=["multimodal","perception"],   context_length=32768,  description="Qianfan OCR+vision"),
]

REPORTING_FREE: List[ORModel] = [
    ORModel("minimax/minimax-m1:free",                is_free=True, capabilities=["communication","reporting"],  context_length=1000000, description="MiniMax M1 — long-context synthesis"),
    ORModel("meta-llama/llama-3.3-70b-instruct:free", is_free=True, capabilities=["communication","reporting","tool_use"], context_length=131072, supports_tools=True, description="Llama 3.3 70B"),
]

LIGHTWEIGHT_FREE: List[ORModel] = [
    ORModel("meta-llama/llama-3.2-3b-instruct:free",  is_free=True, capabilities=["lightweight"],              context_length=131072, description="Llama 3.2 3B edge"),
    ORModel("liquid/lfm-7b:free",                     is_free=True, capabilities=["lightweight","communication"], context_length=32768, description="LFM 7B instruct"),
]

# Auto-routing fallback (OpenRouter picks best available free model)
AUTO_FALLBACK: ORModel = ORModel(
    "openrouter/auto",
    is_free=False,
    capabilities=["reasoning","coding","communication"],
    description="OpenRouter auto-routing",
)

# Flat catalog for easy lookup
ALL_FREE_MODELS: List[ORModel] = (
    REASONING_FREE + CODING_FREE + MULTIMODAL_FREE + REPORTING_FREE + LIGHTWEIGHT_FREE
)

# Capability → ordered model list (preference order)
CAPABILITY_MAP: dict[str, List[ORModel]] = {
    "reasoning":    REASONING_FREE,
    "research":     REASONING_FREE,
    "coding":       CODING_FREE,
    "engineering":  CODING_FREE,
    "multimodal":   MULTIMODAL_FREE,
    "perception":   MULTIMODAL_FREE,
    "communication":REPORTING_FREE,
    "reporting":    REPORTING_FREE,
    "audit":        REPORTING_FREE,
    "lightweight":  LIGHTWEIGHT_FREE,
    "tool_use":     [m for m in REPORTING_FREE if m.supports_tools],
}
