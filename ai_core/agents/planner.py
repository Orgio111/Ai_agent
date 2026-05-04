"""Planner agent: turns a goal into an ordered plan of steps."""
from __future__ import annotations

from typing import Any, Dict, List

from ..logging_setup import logger
from .base import AgentContext, BaseAgent


PLANNER_SYSTEM = """You are PLANNER, the deep-reasoning agent of an autonomous AI system.
You decompose a user goal into 1-6 concrete, ordered steps a downstream Executor can run.

RULES
- Output STRICT JSON only, no prose, no fences.
- Each step has: id (int starting at 1), action (one of: think, tool, code, respond),
  description (one short sentence), tool (string or null), args (object), success_criteria (string).
- Use action='tool' only with a registered tool name. Available tools:
  shell, filesystem, http, rust_perf.
- Use action='code' when source code must be generated (delegate to Coder).
- Use action='respond' for the final answer step.

SCHEMA
{
  "goal": "<echo>",
  "rationale": "<<= 280 chars>",
  "steps": [
    {"id":1,"action":"...","description":"...","tool":null,"args":{},"success_criteria":"..."}
  ]
}
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    tier = "complex"
    system_prompt = PLANNER_SYSTEM

    async def plan(self, ctx: AgentContext) -> Dict[str, Any]:
        prompt = (
            f"GOAL: {ctx.goal}\n\n"
            "Produce a JSON plan now. Ensure final step has action='respond'."
        )
        result = await self.call(ctx, prompt)
        plan = self.extract_json(result["content"])
        if not isinstance(plan, dict) or "steps" not in plan:
            logger.warning("planner returned no parseable JSON; falling back")
            plan = {
                "goal": ctx.goal,
                "rationale": "fallback: direct response",
                "steps": [{
                    "id": 1,
                    "action": "respond",
                    "description": "Answer the goal directly.",
                    "tool": None,
                    "args": {},
                    "success_criteria": "User goal addressed.",
                }],
            }
        plan["_meta"] = {"model": result.get("model"), "tier": result.get("tier")}
        return plan
