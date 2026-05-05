"""Reporter Agent — MiniMax (primary), Gemma + Llama (support).

Handles: CFO-ready reports, executive summaries, dashboards,
data visualization descriptions, structured communication.
"""
from __future__ import annotations

from typing import Any, Dict

from ..logging_setup import logger
from ..model_selector import TaskType, get_selector
from ..openrouter.client import get_openrouter_client
from .base import AgentContext, BaseAgent

_REPORTER_SYSTEM = """\
You are a world-class analyst and communication specialist. You transform complex \
technical analyses into clear, compelling reports for executive stakeholders.

Your reports are:
- Concise yet comprehensive
- Data-driven with specific metrics
- Actionable with clear recommendations
- Formatted in professional markdown
- Structured: Executive Summary → Key Findings → Analysis → Risks → Recommendations

Always adapt the format to the audience (CFO, CTO, board, general).
"""


class ReporterAgent(BaseAgent):
    name = "reporter"
    tier = "balanced"
    system_prompt = _REPORTER_SYSTEM

    def __init__(self, client=None) -> None:
        super().__init__(client)
        self._selector = get_selector()

    async def generate_report(
        self,
        ctx: AgentContext,
        data: str,
        report_type: str = "executive_summary",
        audience: str = "general",
        max_length: str = "medium",
    ) -> Dict[str, Any]:
        """Generate a professional report from raw analysis data."""
        spec, provider = self._selector.select(TaskType.REPORTING)
        logger.info(f"[reporter] model={spec.id} provider={provider} type={report_type}")

        length_guide = {"short": "1-2 paragraphs", "medium": "1-2 pages", "long": "full report"}
        prompt = (
            f"Report type: {report_type}\n"
            f"Audience: {audience}\n"
            f"Length: {length_guide.get(max_length, max_length)}\n\n"
            f"Data to report on:\n{data}\n\n"
            f"Generate a professional {report_type} for a {audience} audience."
        )
        messages = self._build_messages(ctx, prompt)

        if provider == "nim":
            result = await self.client.chat(messages, model=spec.id, temperature=0.4, max_tokens=4096)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=spec.id, temperature=0.4, max_tokens=4096)

        return {
            "report": result.get("content", ""),
            "report_type": report_type,
            "audience": audience,
            "model": spec.id,
            "provider": provider,
        }

    async def cfo_report(self, ctx: AgentContext, financial_data: str) -> Dict[str, Any]:
        """Generate a CFO-ready financial report."""
        return await self.generate_report(
            ctx, financial_data, report_type="financial_summary",
            audience="CFO", max_length="medium"
        )

    async def executive_summary(self, ctx: AgentContext, analysis: str) -> Dict[str, Any]:
        """Generate a board-level executive summary."""
        return await self.generate_report(
            ctx, analysis, report_type="executive_summary",
            audience="board", max_length="short"
        )

    async def technical_report(self, ctx: AgentContext, technical_data: str) -> Dict[str, Any]:
        """Generate a CTO/engineering technical report."""
        return await self.generate_report(
            ctx, technical_data, report_type="technical_analysis",
            audience="CTO", max_length="long"
        )
