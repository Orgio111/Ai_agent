"""Autonomous Goal System — long-horizon task management (8+ hours).

Goals are broken into sub-goals, tracked via a priority queue, and
re-evaluated after each completion cycle. The system persists goal state
so it survives restarts and runs indefinitely.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..logging_setup import logger


class GoalStatus(str, Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    BLOCKED   = "blocked"    # waiting on dependencies
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class GoalPriority(int, Enum):
    CRITICAL = 1
    HIGH     = 2
    MEDIUM   = 3
    LOW      = 4


@dataclass
class Goal:
    id: str
    title: str
    description: str
    priority: int = GoalPriority.MEDIUM
    status: str = GoalStatus.PENDING
    parent_id: Optional[str] = None
    sub_goal_ids: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_overdue(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600.0


class AutonomousGoalSystem:
    """Persistent goal queue with dependency tracking and autonomous execution.

    Supports long-running goals (8+ hours) via background task loop.
    Goal state is persisted to disk so execution survives restarts.
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        poll_interval: float = 5.0,
    ) -> None:
        self._goals: Dict[str, Goal] = {}
        self._persist_path = persist_path
        self._poll_interval = poll_interval
        self._executor: Optional[Callable[[Goal], Coroutine]] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------ #
    # Goal management                                                      #
    # ------------------------------------------------------------------ #

    def add_goal(
        self,
        title: str,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        deadline_hours: Optional[float] = None,
        **meta: Any,
    ) -> Goal:
        goal = Goal(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=int(priority),
            parent_id=parent_id,
            depends_on=depends_on or [],
            deadline=time.time() + deadline_hours * 3600 if deadline_hours else None,
            meta=meta,
        )
        self._goals[goal.id] = goal
        if parent_id and parent_id in self._goals:
            self._goals[parent_id].sub_goal_ids.append(goal.id)
        self._persist()
        logger.info(f"[goals] added: '{goal.title}' (id={goal.id[:8]}, priority={goal.priority})")
        return goal

    def decompose(
        self,
        parent_id: str,
        sub_goals: List[Dict[str, str]],
    ) -> List[Goal]:
        """Break a goal into ordered sub-goals (each depends on the previous)."""
        created: List[Goal] = []
        prev_id: Optional[str] = None
        for sg in sub_goals:
            g = self.add_goal(
                title=sg["title"],
                description=sg.get("description", sg["title"]),
                priority=GoalPriority.HIGH,
                parent_id=parent_id,
                depends_on=[prev_id] if prev_id else [],
            )
            created.append(g)
            prev_id = g.id
        return created

    def get_ready(self) -> List[Goal]:
        """Return goals that are pending and have no unmet dependencies."""
        ready = []
        for g in self._goals.values():
            if g.status != GoalStatus.PENDING:
                continue
            deps_met = all(
                self._goals.get(d, Goal("", "", "")).status == GoalStatus.COMPLETED
                for d in g.depends_on
            )
            if deps_met:
                ready.append(g)
        # Sort by priority (lower int = higher priority), then creation time
        return sorted(ready, key=lambda g: (g.priority, g.created_at))

    def mark_active(self, goal_id: str) -> None:
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.ACTIVE
            self._goals[goal_id].started_at = time.time()
            self._persist()

    def mark_completed(self, goal_id: str, result: str = "") -> None:
        if goal_id in self._goals:
            g = self._goals[goal_id]
            g.status = GoalStatus.COMPLETED
            g.completed_at = time.time()
            g.result = result
            self._persist()
            logger.info(f"[goals] completed: '{g.title}' in {g.age_hours():.2f}h")
            self._check_parent_completion(g)

    def mark_failed(self, goal_id: str, error: str = "") -> None:
        if goal_id in self._goals:
            g = self._goals[goal_id]
            g.retry_count += 1
            if g.retry_count <= g.max_retries:
                g.status = GoalStatus.PENDING  # retry
                logger.warning(f"[goals] retry {g.retry_count}/{g.max_retries}: '{g.title}'")
            else:
                g.status = GoalStatus.FAILED
                g.error = error
                logger.error(f"[goals] failed: '{g.title}' — {error}")
            self._persist()

    def _check_parent_completion(self, goal: Goal) -> None:
        if not goal.parent_id:
            return
        parent = self._goals.get(goal.parent_id)
        if not parent:
            return
        all_done = all(
            self._goals.get(sid, Goal("", "", "")).status == GoalStatus.COMPLETED
            for sid in parent.sub_goal_ids
        )
        if all_done and parent.status == GoalStatus.ACTIVE:
            self.mark_completed(parent.id, result="All sub-goals completed")

    # ------------------------------------------------------------------ #
    # Autonomous execution loop                                            #
    # ------------------------------------------------------------------ #

    def set_executor(self, fn: Callable[[Goal], Coroutine]) -> None:
        """Register the async function that executes a goal."""
        self._executor = fn

    async def start(self) -> None:
        """Start the background goal execution loop."""
        if self._loop_task and not self._loop_task.done():
            return
        self._stop.clear()
        self._loop_task = asyncio.create_task(self._loop())
        logger.info("[goals] autonomous loop started")

    async def stop(self) -> None:
        self._stop.set()
        if self._loop_task:
            await self._loop_task

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ready = self.get_ready()
                if ready and self._executor:
                    # Execute ready goals with limited parallelism
                    batch = ready[:3]  # max 3 concurrent goals
                    await asyncio.gather(*[
                        self._run_goal(g) for g in batch
                    ])
            except Exception as e:
                logger.error(f"[goals] loop error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _run_goal(self, goal: Goal) -> None:
        self.mark_active(goal.id)
        try:
            assert self._executor is not None
            result = await self._executor(goal)
            self.mark_completed(goal.id, result=str(result or ""))
        except Exception as e:
            self.mark_failed(goal.id, error=str(e))

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.write_text(
                json.dumps([asdict(g) for g in self._goals.values()],
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[goals] persist failed: {e}")

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for item in data:
                g = Goal(**{k: v for k, v in item.items() if k in Goal.__dataclass_fields__})
                self._goals[g.id] = g
            logger.info(f"[goals] loaded {len(self._goals)} goals from {self._persist_path}")
        except Exception as e:
            logger.warning(f"[goals] load failed: {e}")

    def status_summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for g in self._goals.values():
            counts[g.status] = counts.get(g.status, 0) + 1
        return {
            "total": len(self._goals),
            "by_status": counts,
            "overdue": sum(1 for g in self._goals.values() if g.is_overdue()),
            "ready": len(self.get_ready()),
        }
