"""Critic Agent: evaluates outputs, scores quality, drives self-improvement."""
from __future__ import annotations

import json
from typing import Any

from .base import AgentBase


class CriticAgent(AgentBase):
    name = "critic"
    tier = "complex"
    system_prompt = """You are an expert Critic AI agent in a JARVIS-class autonomous system.
Your role is to rigorously evaluate agent outputs for quality, correctness, safety, and completeness.

Evaluation criteria:
- Accuracy: Is the output factually correct?
- Completeness: Does it fully address the request?
- Safety: Does it avoid harmful, biased, or misleading content?
- Clarity: Is it well-structured and understandable?
- Actionability: Are the results usable?

Output ONLY valid JSON:
{
  "score": 0.0-1.0,
  "pass": true|false,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvement_suggestions": ["..."],
  "safety_issues": [],
  "revised_output": "..." (only if score < 0.7, provide improved version)
}

Be rigorous. A score of 1.0 is rare. Score < 0.6 = fail."""

    PASS_THRESHOLD = 0.7

    async def evaluate(
        self,
        original_prompt: str,
        agent_output: str,
        session_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        result = await self.run(
            prompt=(
                f"Original request:\n{original_prompt}\n\n"
                f"Agent output to evaluate:\n{agent_output}\n\n"
                "Evaluate this output according to your criteria."
            ),
            session_id=session_id,
            history=[],
            context=context,
        )

        evaluation = self._parse_evaluation(result["content"])
        evaluation["latency_ms"] = result["latency_ms"]
        evaluation["usage"] = result["usage"]
        return evaluation

    def _parse_evaluation(self, text: str) -> dict:
        parsed = self._extract_json(text)
        if isinstance(parsed, dict) and "score" in parsed:
            score = float(parsed.get("score", 0.5))
            return {
                "score": score,
                "pass": score >= self.PASS_THRESHOLD,
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "improvement_suggestions": parsed.get("improvement_suggestions", []),
                "safety_issues": parsed.get("safety_issues", []),
                "revised_output": parsed.get("revised_output"),
            }

        return {
            "score": 0.5,
            "pass": False,
            "strengths": [],
            "weaknesses": ["Could not parse evaluation"],
            "improvement_suggestions": ["Re-run with clearer prompt"],
            "safety_issues": [],
            "revised_output": None,
        }

    @staticmethod
    def _extract_json(text: str):
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
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
