from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class AgentRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    agent_types: Optional[list[str]] = None  # None = run full swarm
    context: dict[str, Any] = {}
    max_iterations: int = 5
    stream: bool = False
    goal_id: Optional[str] = None


class AgentStep(BaseModel):
    agent: str
    input: str
    output: str
    tool_calls: list[dict[str, Any]] = []
    duration_ms: float = 0.0
    tokens: int = 0


class AgentResponse(BaseModel):
    result: str
    session_id: str
    steps: list[AgentStep] = []
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    critic_score: float = 0.0
    iterations: int = 0
    goal_id: Optional[str] = None


class GoalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalCreateRequest(BaseModel):
    description: str
    priority: int = 5
    max_tasks: int = 20
    auto_resume: bool = True
    metadata: dict[str, Any] = {}


class TaskNode(BaseModel):
    task_id: str
    description: str
    depends_on: list[str] = []
    status: str = "pending"
    result: Optional[str] = None
    agent: Optional[str] = None


class GoalStatusResponse(BaseModel):
    goal_id: str
    description: str
    status: GoalStatus
    progress_pct: float = 0.0
    tasks: list[TaskNode] = []
    created_at: str
    updated_at: str
    result: Optional[str] = None
    error: Optional[str] = None


class AgentStatus(BaseModel):
    name: str
    status: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_latency_ms: float = 0.0


class SwarmStatusResponse(BaseModel):
    agents: list[AgentStatus]
    lifecycle: dict[str, Any]
    active_goals: int
