"""Autonomous loop: Planner → Executor → Critic, with self-improvement."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents import (
    AgentContext,
    CoderAgent,
    CriticAgent,
    ExecutorAgent,
    PlannerAgent,
)
from ..config import get_settings
from ..logging_setup import logger
from ..memory import get_memory


@dataclass
class RunResult:
    goal: str
    final_answer: str
    iterations: int
    plan: Dict[str, Any]
    execution: Dict[str, Any]
    critic: Dict[str, Any]
    elapsed_ms: float
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "plan": self.plan,
            "execution": self.execution,
            "critic": self.critic,
            "elapsed_ms": self.elapsed_ms,
            "trace": self.trace,
        }


class Orchestrator:
    """Coordinates the four agents and the memory system."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.memory = get_memory()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.coder = CoderAgent()
        self.critic = CriticAgent()
        self.max_iterations = self.settings.max_iterations
        self.threshold = self.settings.critic_threshold

    async def run(self, goal: str, session_id: str = "default") -> RunResult:
        start = time.perf_counter()
        trace: List[Dict[str, Any]] = []

        ctx = AgentContext(session_id=session_id, goal=goal)
        ctx.history = self.memory.history(session_id)
        try:
            ctx.memory_snippets = await self.memory.recall(goal, k=4)
        except Exception as e:  # embedding may fail without API key in tests
            logger.warning(f"recall failed: {e}")
            ctx.memory_snippets = ""

        self.memory.add_message(session_id, "user", goal)

        plan: Dict[str, Any] = {}
        execution: Dict[str, Any] = {}
        verdict: Dict[str, Any] = {}
        iterations = 0

        for i in range(1, self.max_iterations + 1):
            iterations = i
            logger.info(f"iteration {i}: planning")
            plan = await self.planner.plan(ctx)
            trace.append({"phase": "plan", "iteration": i, "data": plan})

            logger.info(f"iteration {i}: executing {len(plan.get('steps', []))} steps")
            execution = await self.executor.execute_plan(ctx, plan)
            trace.append({"phase": "execute", "iteration": i, "data": execution})

            logger.info(f"iteration {i}: critiquing")
            verdict = await self.critic.review(ctx, plan, execution)
            trace.append({"phase": "critique", "iteration": i, "data": verdict})

            if verdict.get("verdict") == "accept" or verdict.get("score", 0.0) >= self.threshold:
                break

            # Refine: feed suggestions back into goal context.
            suggestions = verdict.get("suggestions", [])
            if not suggestions:
                break
            ctx.goal = (
                f"{goal}\n\nPrior attempt scored {verdict.get('score'):.2f}. "
                f"Address: {'; '.join(suggestions)}"
            )

        final = execution.get("final", "") if execution else ""
        self.memory.add_message(session_id, "assistant", final)

        # Persist a long-term trace summary.
        try:
            await self.memory.store(
                text=f"Q: {goal}\nA: {final}",
                tags=["dialogue", session_id],
                meta={"score": verdict.get("score", 0.0)},
            )
        except Exception as e:
            logger.warning(f"long-term store failed: {e}")

        elapsed = (time.perf_counter() - start) * 1000.0

        return RunResult(
            goal=goal,
            final_answer=final,
            iterations=iterations,
            plan=plan,
            execution=execution,
            critic=verdict,
            elapsed_ms=elapsed,
            trace=trace if self.settings.raw_settings.get("orchestrator", {}).get("trace", True) else [],
        )


_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


class AgentOrchestrator:
    """Test-compatible orchestrator that accepts an AgentContext directly."""

    def __init__(self) -> None:
        self._orch = Orchestrator()

    async def run(self, ctx: AgentContext) -> RunResult:
        return await self._orch.run(goal=ctx.goal, session_id=ctx.session_id)
