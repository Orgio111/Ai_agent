"""Autonomous loop: Classify → Select → Planner → Executor → Audit → Critic.

Integrates:
  - TaskClassifier (keyword-based task routing)
  - 3-tier ModelSelector (NIM → OpenRouter → Ranked fallback)
  - SelfImprovingLoop (performance tracking + adaptation)
  - AutonomousGoalSystem (long-horizon goal management)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents import (
    AgentContext,
    AuditAgent,
    CoderAgent,
    CriticAgent,
    EngineerAgent,
    ExecutorAgent,
    PerceptionAgent,
    PlannerAgent,
    ReporterAgent,
    ResearchAgent,
)
from ..config import get_settings
from ..goals.system import AutonomousGoalSystem
from ..logging_setup import logger
from ..memory import get_memory
from ..model_selector import get_selector
from ..multi_model import get_executor as get_multi_executor
from ..orchestrator.task_classifier import get_classifier
from ..self_improve.loop import SelfImprovingLoop


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
    """Coordinates agents, memory, task classification, and self-improvement."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.memory = get_memory()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.coder = CoderAgent()
        self.critic = CriticAgent()
        # Specialized agents (3-tier model selection)
        self.perception = PerceptionAgent()
        self.researcher = ResearchAgent()
        self.engineer = EngineerAgent()
        self.auditor = AuditAgent()
        self.reporter = ReporterAgent()
        # Routing + improvement
        self.classifier = get_classifier()
        self.selector = get_selector()
        self.multi_executor = get_multi_executor()
        data_dir = self.settings.memory_dir.parent / "system"
        self.goal_system = AutonomousGoalSystem(
            persist_path=data_dir / "goals.json"
        )
        self.self_improve = SelfImprovingLoop(
            persist_path=data_dir / "performance.json"
        )
        self.max_iterations = self.settings.max_iterations
        self.threshold = self.settings.critic_threshold

    async def run(self, goal: str, session_id: str = "default") -> RunResult:  # noqa: C901
        start = time.perf_counter()
        trace: List[Dict[str, Any]] = []

        # Classify task type for intelligent model routing
        task_type = self.classifier.classify(goal)
        logger.info(f"[orchestrator] task_type={task_type.value} goal={goal[:60]}...")

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

        # Record performance for self-improvement (critic score as quality proxy)
        quality = float(verdict.get("score", 0.5))
        self.self_improve.record(
            model_id=self.settings.models.routing.get("balanced", None) and
                     self.settings.models.routing["balanced"].model or "unknown",
            task_type=task_type,
            quality_score=quality,
            latency_ms=(time.perf_counter() - start) * 1000,
            success=verdict.get("verdict") == "accept" or quality >= self.threshold,
        )

        # Fire-and-forget long-term persistence — doesn't block returning the result.
        asyncio.create_task(self._store_async(goal, final, verdict, session_id))

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


    async def _store_async(
        self, goal: str, final: str, verdict: Dict[str, Any], session_id: str
    ) -> None:
        try:
            await self.memory.store(
                text=f"Q: {goal}\nA: {final}",
                tags=["dialogue", session_id],
                meta={"score": verdict.get("score", 0.0)},
            )
        except Exception as e:
            logger.warning(f"long-term store failed: {e}")


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
