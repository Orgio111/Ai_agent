"""Planner Agent: breaks goals into structured task DAGs."""
from __future__ import annotations

import json
from typing import Any

from .base import AgentBase


class PlannerAgent(AgentBase):
    name = "planner"
    tier = "complex"
    system_prompt = """You are an expert Planner AI agent in a JARVIS-class autonomous system.

Your role:
1. Analyze the user's goal or request thoroughly
2. Decompose it into a structured execution plan
3. Identify which agent should handle each task (executor, researcher, critic, optimizer)
4. Specify task dependencies as a DAG
5. Estimate complexity and resource requirements

Output ONLY valid JSON:
{
  "plan_summary": "...",
  "tasks": [
    {
      "task_id": "t1",
      "description": "...",
      "agent": "executor|researcher|optimizer",
      "depends_on": [],
      "priority": 1-5,
      "estimated_tokens": 500
    }
  ],
  "success_criteria": "...",
  "fallback_strategy": "..."
}

If the request is simple, produce a single task. Be precise and actionable."""

    async def plan(
        self,
        goal: str,
        session_id: str,
        history: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        result = await self.run(
            prompt=f"Create a detailed execution plan for: {goal}",
            session_id=session_id,
            history=history,
            context=context,
        )

        content = result["content"]
        parsed = self._safe_parse_plan(content)

        return {
            "plan": parsed,
            "raw_content": content,
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
        }

    def _safe_parse_plan(self, text: str) -> dict:
        parsed = self.extract_json(text)
        if isinstance(parsed, dict) and "tasks" in parsed:
            return parsed

        return {
            "plan_summary": text[:200],
            "tasks": [
                {
                    "task_id": "t1",
                    "description": text[:500],
                    "agent": "executor",
                    "depends_on": [],
                    "priority": 3,
                    "estimated_tokens": 1000,
                }
            ],
            "success_criteria": "Task completed without errors",
            "fallback_strategy": "Retry with simpler approach",
        }

    @staticmethod
    def extract_json(text: str):
        import re
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None
