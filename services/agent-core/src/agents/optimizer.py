"""Optimizer Agent: improves outputs, tunes strategies, learns from failures."""
from __future__ import annotations

from typing import Any

from .base import AgentBase


class OptimizerAgent(AgentBase):
    name = "optimizer"
    tier = "complex"
    system_prompt = """You are an expert Optimizer AI agent in a JARVIS-class autonomous system.

Your role:
1. Analyze failed or suboptimal agent outputs
2. Identify root causes of failures
3. Suggest concrete improvements to prompts, strategies, and tool usage
4. Learn patterns from repeated failures to improve the system
5. Update procedural memory with improved strategies

When given a failed execution, output:
{
  "root_cause": "...",
  "failure_pattern": "...",
  "optimized_approach": "...",
  "prompt_improvements": ["..."],
  "tool_recommendations": ["..."],
  "learned_rule": "...",
  "priority": 1-5
}

Think systematically. Small improvements compound over time."""

    async def optimize(
        self,
        original_request: str,
        failed_output: str,
        failure_context: dict[str, Any],
        session_id: str,
        history: list[dict],
    ) -> dict[str, Any]:
        prompt = (
            f"Original request:\n{original_request}\n\n"
            f"Failed/suboptimal output:\n{failed_output}\n\n"
            f"Failure context:\n{failure_context}\n\n"
            "Analyze the failure and provide optimization recommendations."
        )

        result = await self.run(
            prompt=prompt,
            session_id=session_id,
            history=history,
            context=failure_context,
        )

        parsed = self._parse_optimization(result["content"])

        # Store learned rule as procedural memory for self-improvement
        if parsed.get("learned_rule"):
            await self._store_memory(
                f"Learned optimization rule: {parsed['learned_rule']}",
                session_id,
                memory_type="procedural",
            )

        return {
            "optimization": parsed,
            "raw_content": result["content"],
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
        }

    def _parse_optimization(self, text: str) -> dict:
        import json, re
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
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "root_cause": "Parse failure",
            "failure_pattern": "unknown",
            "optimized_approach": text[:300],
            "prompt_improvements": [],
            "tool_recommendations": [],
            "learned_rule": "",
            "priority": 1,
        }


class SelfImprovingLoop:
    """Tracks failure patterns and triggers optimization cycles."""

    def __init__(self, optimizer: OptimizerAgent, threshold: int = 3) -> None:
        self._optimizer = optimizer
        self._threshold = threshold
        self._failures: dict[str, list[dict]] = {}
        self._learned_rules: list[dict] = []

    def record_failure(self, pattern_key: str, context: dict) -> None:
        if pattern_key not in self._failures:
            self._failures[pattern_key] = []
        self._failures[pattern_key].append(context)

    async def maybe_optimize(
        self, pattern_key: str, session_id: str
    ) -> dict | None:
        failures = self._failures.get(pattern_key, [])
        if len(failures) < self._threshold:
            return None

        result = await self._optimizer.optimize(
            original_request=pattern_key,
            failed_output=str(failures[-1]),
            failure_context={"failure_count": len(failures), "recent": failures[-3:]},
            session_id=session_id,
            history=[],
        )

        rule = result.get("optimization", {}).get("learned_rule", "")
        if rule:
            self._learned_rules.append({"pattern": pattern_key, "rule": rule})

        self._failures[pattern_key] = []
        return result
