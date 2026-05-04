"""Autonomous Goal Engine: long-running goals, DAG task execution,
interrupt/resume, multi-goal scheduling."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import logging

from .models import (
    GoalCreateRequest,
    GoalStatus,
    GoalStatusResponse,
    TaskNode,
)

logger = logging.getLogger(__name__)


class GoalEngine:
    def __init__(self, swarm, memory_url: str) -> None:
        self._swarm = swarm
        self._memory_url = memory_url
        self._goals: dict[str, dict] = {}
        self._running_goals: dict[str, asyncio.Task] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._semaphore = asyncio.Semaphore(3)  # max concurrent goals

    async def start(self) -> None:
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Goal engine started")

    async def stop(self) -> None:
        self._running = False
        for task in self._running_goals.values():
            task.cancel()
        if self._scheduler_task:
            self._scheduler_task.cancel()

    async def create(self, req: GoalCreateRequest) -> GoalStatusResponse:
        goal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        goal = {
            "goal_id": goal_id,
            "description": req.description,
            "status": GoalStatus.PENDING,
            "priority": req.priority,
            "max_tasks": req.max_tasks,
            "auto_resume": req.auto_resume,
            "metadata": req.metadata,
            "tasks": [],
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
        self._goals[goal_id] = goal
        logger.info(f"Goal created: {goal_id} — {req.description[:60]}")

        # Immediately start if capacity available
        asyncio.create_task(self._maybe_start_goal(goal_id))
        return self._to_response(goal)

    async def get(self, goal_id: str) -> Optional[GoalStatusResponse]:
        goal = self._goals.get(goal_id)
        return self._to_response(goal) if goal else None

    async def cancel(self, goal_id: str) -> None:
        goal = self._goals.get(goal_id)
        if not goal:
            return
        goal["status"] = GoalStatus.CANCELLED
        goal["updated_at"] = datetime.now(timezone.utc).isoformat()
        if goal_id in self._running_goals:
            self._running_goals[goal_id].cancel()
            del self._running_goals[goal_id]

    async def list_all(self) -> list[dict]:
        return [self._to_response(g).model_dump() for g in self._goals.values()]

    async def active_count(self) -> int:
        return len([g for g in self._goals.values() if g["status"] == GoalStatus.RUNNING])

    async def _maybe_start_goal(self, goal_id: str) -> None:
        goal = self._goals.get(goal_id)
        if not goal or goal["status"] != GoalStatus.PENDING:
            return
        async with self._semaphore:
            task = asyncio.create_task(self._execute_goal(goal_id))
            self._running_goals[goal_id] = task
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Goal {goal_id} cancelled")
            finally:
                self._running_goals.pop(goal_id, None)

    async def _execute_goal(self, goal_id: str) -> None:
        goal = self._goals[goal_id]
        goal["status"] = GoalStatus.RUNNING
        goal["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            logger.info(f"Executing goal {goal_id}: {goal['description'][:60]}")

            # Step 1: Plan the goal into tasks
            from .models import AgentRequest
            plan_req = AgentRequest(
                prompt=f"Create a detailed plan to accomplish: {goal['description']}",
                session_id=f"goal_{goal_id}",
                agent_types=["planner"],
            )
            assert self._swarm.planner is not None
            plan_result = await self._swarm.planner.plan(
                goal["description"],
                session_id=f"goal_{goal_id}",
                history=[],
                context=goal.get("metadata", {}),
            )
            tasks = plan_result["plan"].get("tasks", [])[:goal["max_tasks"]]

            goal["tasks"] = [
                {
                    "task_id": t.get("task_id", f"t{i}"),
                    "description": t.get("description", ""),
                    "depends_on": t.get("depends_on", []),
                    "status": "pending",
                    "result": None,
                    "agent": t.get("agent", "executor"),
                }
                for i, t in enumerate(tasks)
            ]

            # Step 2: Execute tasks respecting DAG
            completed: dict[str, str] = {}
            total = len(goal["tasks"])

            for task_def in self._topological_sort(goal["tasks"]):
                task_id = task_def["task_id"]

                # Check if cancelled
                goal_state = self._goals.get(goal_id, {})
                if goal_state.get("status") == GoalStatus.CANCELLED:
                    return

                task_def["status"] = "running"
                goal["updated_at"] = datetime.now(timezone.utc).isoformat()

                assert self._swarm.executor is not None
                try:
                    exec_result = await self._swarm.executor.execute(
                        task=task_def,
                        session_id=f"goal_{goal_id}",
                        history=[],
                        context={**goal.get("metadata", {}), "completed_tasks": completed},
                    )
                    task_def["status"] = "completed"
                    task_def["result"] = exec_result["content"][:500]
                    completed[task_id] = exec_result["content"]
                except Exception as e:
                    task_def["status"] = "failed"
                    task_def["result"] = str(e)
                    logger.error(f"Goal {goal_id} task {task_id} failed: {e}")

                progress = len(completed) / max(total, 1) * 100
                logger.debug(f"Goal {goal_id} progress: {progress:.0f}%")

            # Step 3: Synthesize final result
            final = "\n\n".join(
                f"**{t['task_id']}**: {t.get('result', 'no result')}"
                for t in goal["tasks"]
                if t["status"] == "completed"
            )
            goal["result"] = final or "No tasks completed"
            goal["status"] = GoalStatus.COMPLETED

            # Store in semantic memory
            await self._store_goal_memory(goal)
            logger.info(f"Goal {goal_id} completed successfully")

        except asyncio.CancelledError:
            goal["status"] = GoalStatus.CANCELLED
            raise
        except Exception as e:
            goal["status"] = GoalStatus.FAILED
            goal["error"] = str(e)
            logger.error(f"Goal {goal_id} failed: {e}")

            # Auto-resume if configured
            if goal.get("auto_resume") and goal.get("status") != GoalStatus.CANCELLED:
                logger.info(f"Auto-resuming goal {goal_id}")
                goal["status"] = GoalStatus.PENDING
                asyncio.create_task(self._maybe_start_goal(goal_id))

        goal["updated_at"] = datetime.now(timezone.utc).isoformat()

    async def _store_goal_memory(self, goal: dict) -> None:
        try:
            async with httpx.AsyncClient(base_url=self._memory_url, timeout=5.0) as client:
                await client.post(
                    "/store",
                    json={
                        "content": f"Completed goal '{goal['description'][:200]}': {goal.get('result', '')[:300]}",
                        "memory_type": "episodic",
                        "importance": 0.8,
                        "metadata": {"goal_id": goal["goal_id"]},
                    },
                )
        except Exception:
            pass

    def _topological_sort(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        task_map = {t["task_id"]: t for t in tasks}
        in_degree = {t["task_id"]: len([d for d in t.get("depends_on", []) if d in task_map]) for t in tasks}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            tid = queue.pop(0)
            if tid in task_map:
                result.append(task_map[tid])
            for t in tasks:
                if tid in t.get("depends_on", []):
                    in_degree[t["task_id"]] -= 1
                    if in_degree[t["task_id"]] == 0:
                        queue.append(t["task_id"])
        remaining = [t for t in tasks if t not in result]
        return result + remaining

    async def _scheduler_loop(self) -> None:
        while self._running:
            await asyncio.sleep(5)
            pending = [
                g for g in self._goals.values()
                if g["status"] == GoalStatus.PENDING
                and g["goal_id"] not in self._running_goals
            ]
            pending.sort(key=lambda g: g.get("priority", 5), reverse=True)
            for goal in pending[:2]:
                asyncio.create_task(self._maybe_start_goal(goal["goal_id"]))

    def _to_response(self, goal: dict) -> GoalStatusResponse:
        tasks = [
            TaskNode(
                task_id=t["task_id"],
                description=t["description"],
                depends_on=t.get("depends_on", []),
                status=t.get("status", "pending"),
                result=t.get("result"),
                agent=t.get("agent"),
            )
            for t in goal.get("tasks", [])
        ]
        completed = len([t for t in tasks if t.status == "completed"])
        total = len(tasks)
        return GoalStatusResponse(
            goal_id=goal["goal_id"],
            description=goal["description"],
            status=goal["status"],
            progress_pct=(completed / max(total, 1)) * 100,
            tasks=tasks,
            created_at=goal["created_at"],
            updated_at=goal["updated_at"],
            result=goal.get("result"),
            error=goal.get("error"),
        )
