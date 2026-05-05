"""Critic agent: scores and critiques the executor's output."""
from __future__ import annotations

from typing import Any, Dict

from ..logging_setup import logger
from .base import AgentContext, BaseAgent

CRITIC_SYSTEM = """You are CRITIC, a strict reviewer.
You evaluate whether the executor's final answer satisfies the user goal.

OUTPUT STRICT JSON ONLY:
{
  "score": <0.0-1.0>,
  "verdict": "accept" | "improve",
  "issues": ["..."],
  "suggestions": ["..."]
}
- score >= 0.75 → verdict='accept'
- score < 0.75  → verdict='improve' and provide concrete suggestions
"""


class CriticAgent(BaseAgent):
    name = "critic"
    tier = "balanced"
    system_prompt = CRITIC_SYSTEM

    async def review(self, ctx: AgentContext, plan: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"GOAL:\n{ctx.goal}\n\n"
            f"PLAN RATIONALE:\n{plan.get('rationale', '')}\n\n"
            f"FINAL ANSWER:\n{result.get('final', '')}\n\n"
            "Judge it now. JSON only."
        )
        res = await self.call(ctx, prompt)
        verdict = self.extract_json(res["content"])
        if not isinstance(verdict, dict):
            logger.warning("critic returned no JSON; defaulting to accept")
            verdict = {"score": 0.8, "verdict": "accept", "issues": [], "suggestions": []}
        # Normalize.
        verdict.setdefault("issues", [])
        verdict.setdefault("suggestions", [])
        try:
            verdict["score"] = float(verdict.get("score", 0.0))
        except (TypeError, ValueError):
            verdict["score"] = 0.0
        verdict["verdict"] = "accept" if verdict["score"] >= 0.75 else verdict.get("verdict", "improve")
        return verdict
