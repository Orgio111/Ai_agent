"""Model capability taxonomy and unified model spec."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Capability(str, Enum):
    REASONING    = "reasoning"
    RESEARCH     = "research"
    CODING       = "coding"
    ENGINEERING  = "engineering"
    MULTIMODAL   = "multimodal"
    PERCEPTION   = "perception"
    COMMUNICATION = "communication"
    REPORTING    = "reporting"
    AUDIT        = "audit"
    LIGHTWEIGHT  = "lightweight"
    TOOL_USE     = "tool_use"
    PLANNING     = "planning"


class TaskType(str, Enum):
    """High-level task categories used by the task classifier."""
    PERCEPTION   = "perception"    # Image/doc/audio understanding
    RESEARCH     = "research"      # Deep search + analysis
    ENGINEERING  = "engineering"   # Code gen / execution
    AUDIT        = "audit"         # Validation / compliance
    REPORTING    = "reporting"     # Synthesis / communication
    PLANNING     = "planning"      # Orchestration / strategy
    REASONING    = "reasoning"     # Multi-step logic
    GENERAL      = "general"       # Catch-all


# Which capabilities satisfy each task type
TASK_CAPABILITY_MAP: dict[TaskType, List[Capability]] = {
    TaskType.PERCEPTION:  [Capability.PERCEPTION, Capability.MULTIMODAL],
    TaskType.RESEARCH:    [Capability.RESEARCH, Capability.REASONING],
    TaskType.ENGINEERING: [Capability.ENGINEERING, Capability.CODING, Capability.TOOL_USE],
    TaskType.AUDIT:       [Capability.AUDIT, Capability.REASONING, Capability.TOOL_USE],
    TaskType.REPORTING:   [Capability.REPORTING, Capability.COMMUNICATION],
    TaskType.PLANNING:    [Capability.PLANNING, Capability.REASONING],
    TaskType.REASONING:   [Capability.REASONING, Capability.RESEARCH],
    TaskType.GENERAL:     [Capability.REASONING, Capability.COMMUNICATION],
}


@dataclass
class UnifiedModelSpec:
    """Provider-agnostic model descriptor used by ModelSelector."""
    id: str
    provider: str           # "nim" | "openrouter" | "ranked"
    tier: int               # 1 = NIM, 2 = OR free, 3 = ranked fallback
    capabilities: List[Capability]
    max_context: int = 32768
    supports_tools: bool = False
    cost_per_token: float = 0.0
    is_free: bool = False
    available: bool = True
    description: str = ""

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def supports_task(self, task: TaskType) -> bool:
        required = TASK_CAPABILITY_MAP.get(task, [])
        return any(self.supports(c) for c in required)
