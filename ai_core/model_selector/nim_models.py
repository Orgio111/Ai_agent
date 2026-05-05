"""LEVEL 1 — NVIDIA NIM core model definitions (5 specialist models).

Each model maps to a primary agent role in the system:

  1. Nemotron Super    → Strategic Planner / Orchestrator
  2. Nemotron Nano Omni→ Multimodal Perception
  3. DeepSeek R1       → Deep Reasoning / Research
  4. GPT-OSS-120B      → Audit / Verification (large-OSS equivalent)
  5. GLM / Qwen Eng    → Engineering / Autonomous Execution
"""
from __future__ import annotations

import os
from typing import List

from .capability import Capability, TaskType, UnifiedModelSpec

# Allow per-deployment overrides via env vars.
_NEMOTRON_SUPER = os.getenv(
    "NIM_MODEL_PLANNER", "nvidia/llama-3.1-nemotron-ultra-253b-v1"
)
_NEMOTRON_NANO = os.getenv(
    "NIM_MODEL_PERCEPTION", "nvidia/nemotron-mini-4b-instruct"
)
_DEEPSEEK_R1 = os.getenv(
    "NIM_MODEL_RESEARCHER", "deepseek-ai/deepseek-r1"
)
_GPT_OSS_120B = os.getenv(
    "NIM_MODEL_AUDITOR", "meta/llama-3.1-405b-instruct"
)
_GLM_ENG = os.getenv(
    "NIM_MODEL_ENGINEER", "qwen/qwen2.5-72b-instruct"
)


NIM_CORE_MODELS: List[UnifiedModelSpec] = [
    UnifiedModelSpec(
        id=_NEMOTRON_SUPER,
        provider="nim",
        tier=1,
        capabilities=[
            Capability.PLANNING, Capability.REASONING, Capability.RESEARCH,
            Capability.COMMUNICATION,
        ],
        max_context=128000,
        supports_tools=True,
        description="Nemotron Super — strategic planner / long-context orchestrator",
    ),
    UnifiedModelSpec(
        id=_NEMOTRON_NANO,
        provider="nim",
        tier=1,
        capabilities=[
            Capability.PERCEPTION, Capability.MULTIMODAL, Capability.LIGHTWEIGHT,
        ],
        max_context=32768,
        supports_tools=False,
        description="Nemotron Nano Omni — multimodal perception",
    ),
    UnifiedModelSpec(
        id=_DEEPSEEK_R1,
        provider="nim",
        tier=1,
        capabilities=[
            Capability.REASONING, Capability.RESEARCH, Capability.PLANNING,
        ],
        max_context=131072,
        supports_tools=True,
        description="DeepSeek R1 — deep chain-of-thought reasoning / research",
    ),
    UnifiedModelSpec(
        id=_GPT_OSS_120B,
        provider="nim",
        tier=1,
        capabilities=[
            Capability.AUDIT, Capability.REASONING, Capability.TOOL_USE,
            Capability.COMMUNICATION,
        ],
        max_context=131072,
        supports_tools=True,
        description="Large OSS 120B+ — audit / verification / hallucination filtering",
    ),
    UnifiedModelSpec(
        id=_GLM_ENG,
        provider="nim",
        tier=1,
        capabilities=[
            Capability.ENGINEERING, Capability.CODING, Capability.TOOL_USE,
        ],
        max_context=131072,
        supports_tools=True,
        description="Engineering LLM — code generation / backtesting / autonomous execution",
    ),
]

# Role → primary NIM model
NIM_ROLE_MAP: dict[str, UnifiedModelSpec] = {
    "planner":    NIM_CORE_MODELS[0],
    "perception": NIM_CORE_MODELS[1],
    "researcher": NIM_CORE_MODELS[2],
    "auditor":    NIM_CORE_MODELS[3],
    "engineer":   NIM_CORE_MODELS[4],
}

# TaskType → preferred NIM model
NIM_TASK_MAP: dict[TaskType, UnifiedModelSpec] = {
    TaskType.PLANNING:    NIM_CORE_MODELS[0],
    TaskType.PERCEPTION:  NIM_CORE_MODELS[1],
    TaskType.RESEARCH:    NIM_CORE_MODELS[2],
    TaskType.REASONING:   NIM_CORE_MODELS[2],
    TaskType.AUDIT:       NIM_CORE_MODELS[3],
    TaskType.ENGINEERING: NIM_CORE_MODELS[4],
    TaskType.REPORTING:   NIM_CORE_MODELS[0],
    TaskType.GENERAL:     NIM_CORE_MODELS[2],
}
