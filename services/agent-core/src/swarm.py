"""Agent Swarm: orchestrates parallel multi-agent execution with speculative
branching, self-correction, and priority scheduling."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

import httpx

from .agents import (
    CriticAgent,
    OptimizerAgent,
    PlannerAgent,
    ResearchAgent,
    SelfImprovingLoop,
    SmartExecutorAgent,
)
from .models import AgentRequest, AgentResponse, AgentStatus, AgentStep

logger = logging.getLogger(__name__)

CRITIC_THRESHOLD = 0.7
MAX_CRITIC_CYCLES = 2


class AgentSwarm:
    def __init__(
        self,
        nim_api_key: str,
        nim_base_url: str,
        memory_url: str,
        tool_url: str,
    ) -> None:
        self._nim_key = nim_api_key
        self._nim_base = nim_base_url
        self._memory_url = memory_url
        self._tool_url = tool_url
        self._nim_client: Optional[httpx.AsyncClient] = None

        self.planner: Optional[PlannerAgent] = None
        self.executor: Optional[SmartExecutorAgent] = None
        self.critic: Optional[CriticAgent] = None
        self.researcher: Optional[ResearchAgent] = None
        self.optimizer: Optional[OptimizerAgent] = None
        self._self_improve: Optional[SelfImprovingLoop] = None

        # Session histories
        self._histories: dict[str, list[dict]] = {}

    async def start(self) -> None:
        llm_url = "http://localhost:8002"  # LLM engine
        self._nim_client = httpx.AsyncClient(
            base_url=llm_url,
            headers={"Authorization": f"Bearer {self._nim_key}"},
            timeout=120.0,
        )

        args = (self._nim_client, self._memory_url, self._tool_url, self._nim_base)
        self.planner = PlannerAgent(*args)
        self.executor = SmartExecutorAgent(*args)
        self.critic = CriticAgent(*args)
        self.researcher = ResearchAgent(*args)
        self.optimizer = OptimizerAgent(*args)
        self._self_improve = SelfImprovingLoop(self.optimizer)

        logger.info("Agent swarm initialized")

    async def stop(self) -> None:
        if self._nim_client:
            await self._nim_client.aclose()

    async def run(self, req: AgentRequest) -> AgentResponse:
        assert all([self.planner, self.executor, self.critic, self.researcher, self.optimizer])

        start = time.monotonic()
        session_id = req.session_id
        history = self._histories.get(session_id, [])
        steps: list[AgentStep] = []
        total_tokens = 0

        # 1. Planner: decompose goal
        logger.info(f"[{session_id}] Planning: {req.prompt[:80]}...")
        plan_result = await self.planner.plan(req.prompt, session_id, history, req.context)
        steps.append(
            AgentStep(
                agent="planner",
                input=req.prompt,
                output=json.dumps(plan_result["plan"]),
                duration_ms=plan_result["latency_ms"],
                tokens=plan_result["usage"].get("total_tokens", 0),
            )
        )
        total_tokens += plan_result["usage"].get("total_tokens", 0)
        plan = plan_result["plan"]
        tasks = plan.get("tasks", [])

        # 2. Research phase (parallel for research tasks)
        research_tasks = [t for t in tasks if t.get("agent") == "researcher"]
        if research_tasks:
            research_results = await asyncio.gather(
                *[
                    self.researcher.research(t["description"], session_id, history, req.context)
                    for t in research_tasks
                ]
            )
            for t, r in zip(research_tasks, research_results):
                steps.append(
                    AgentStep(
                        agent="researcher",
                        input=t["description"],
                        output=r["content"][:500],
                        duration_ms=r["latency_ms"],
                        tokens=r["usage"].get("total_tokens", 0),
                    )
                )
                total_tokens += r["usage"].get("total_tokens", 0)

        # 3. Execute tasks (respecting DAG dependencies)
        execution_results: dict[str, str] = {}
        exec_tasks = [t for t in tasks if t.get("agent") != "researcher"]

        for task in self._topological_sort(exec_tasks):
            task_context = {**req.context, "completed_tasks": execution_results}
            exec_result = await self.executor.execute(
                task=task,
                session_id=session_id,
                history=history,
                context=task_context,
            )
            execution_results[task["task_id"]] = exec_result["content"]
            steps.append(
                AgentStep(
                    agent="executor",
                    input=task["description"],
                    output=exec_result["content"][:500],
                    tool_calls=exec_result.get("tool_calls", []),
                    duration_ms=exec_result["latency_ms"],
                    tokens=exec_result["usage"].get("total_tokens", 0),
                )
            )
            total_tokens += exec_result["usage"].get("total_tokens", 0)

        # 4. Synthesize final result
        final_result = self._synthesize(req.prompt, execution_results, plan)

        # 5. Critic loop (highest priority)
        critic_score = 0.0
        for cycle in range(MAX_CRITIC_CYCLES):
            eval_result = await self.critic.evaluate(
                req.prompt, final_result, session_id, req.context
            )
            critic_score = eval_result["score"]
            steps.append(
                AgentStep(
                    agent="critic",
                    input=final_result[:200],
                    output=f"score={critic_score:.2f} pass={eval_result['pass']}",
                    duration_ms=eval_result.get("latency_ms", 0),
                    tokens=eval_result.get("usage", {}).get("total_tokens", 0),
                )
            )
            total_tokens += eval_result.get("usage", {}).get("total_tokens", 0)

            if eval_result["pass"]:
                break

            # Use revised output if critic provided one
            if eval_result.get("revised_output"):
                final_result = eval_result["revised_output"]
            else:
                # Trigger optimizer for self-improvement
                opt_result = await self.optimizer.optimize(
                    original_request=req.prompt,
                    failed_output=final_result,
                    failure_context={"critic_score": critic_score, "weaknesses": eval_result["weaknesses"]},
                    session_id=session_id,
                    history=history,
                )
                optimized = opt_result.get("optimization", {})
                if optimized.get("optimized_approach"):
                    final_result = optimized["optimized_approach"]
                steps.append(
                    AgentStep(
                        agent="optimizer",
                        input=str(eval_result["weaknesses"]),
                        output=optimized.get("learned_rule", "optimized"),
                        duration_ms=opt_result.get("latency_ms", 0),
                        tokens=opt_result.get("usage", {}).get("total_tokens", 0),
                    )
                )

        # Update session history
        history.append({"role": "user", "content": req.prompt})
        history.append({"role": "assistant", "content": final_result})
        self._histories[session_id] = history[-40:]

        total_ms = (time.monotonic() - start) * 1000
        return AgentResponse(
            result=final_result,
            session_id=session_id,
            steps=steps,
            total_duration_ms=total_ms,
            total_tokens=total_tokens,
            critic_score=critic_score,
            iterations=len([s for s in steps if s.agent == "executor"]),
            goal_id=req.goal_id,
        )

    async def run_stream(self, req: AgentRequest) -> AsyncIterator[str]:
        session_id = req.session_id
        history = self._histories.get(session_id, [])
        assert self.executor is not None

        yield json.dumps({"agent": "executor", "status": "starting"})
        async for chunk in self.executor.run_stream(
            req.prompt, session_id, history, req.context
        ):
            yield chunk

    def agent_statuses(self) -> list[AgentStatus]:
        agents = [
            ("planner", self.planner),
            ("executor", self.executor),
            ("critic", self.critic),
            ("researcher", self.researcher),
            ("optimizer", self.optimizer),
        ]
        return [
            AgentStatus(
                name=name,
                status="active" if agent is not None else "inactive",
                tasks_completed=agent.tasks_completed if agent else 0,
                tasks_failed=agent.tasks_failed if agent else 0,
                avg_latency_ms=agent.avg_latency_ms if agent else 0.0,
            )
            for name, agent in agents
        ]

    def _topological_sort(self, tasks: list[dict]) -> list[dict]:
        """Kahn's algorithm for DAG topological sort."""
        if not tasks:
            return []
        task_map = {t["task_id"]: t for t in tasks}
        in_degree = {t["task_id"]: 0 for t in tasks}
        for t in tasks:
            for dep in t.get("depends_on", []):
                if dep in in_degree:
                    in_degree[t["task_id"]] += 1

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

        # Append any remaining tasks (in case of cycles)
        remaining = [t for t in tasks if t not in result]
        return result + remaining

    def _synthesize(
        self, prompt: str, results: dict[str, str], plan: dict
    ) -> str:
        if len(results) == 1:
            return list(results.values())[0]
        parts = [f"# Result for: {prompt[:100]}\n"]
        for task_id, content in results.items():
            parts.append(f"## {task_id}\n{content}\n")
        return "\n".join(parts)
