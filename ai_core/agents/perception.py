"""Perception Agent — Nemotron Nano Omni (primary), multimodal swarm (support).

Handles: chart reading, PDF parsing, image understanding, data structuring.
Uses 3-tier model selection with multimodal capability requirement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..logging_setup import logger
from ..model_selector import TaskType, get_selector
from ..openrouter.client import get_openrouter_client
from .base import AgentContext, BaseAgent


class PerceptionAgent(BaseAgent):
    name = "perception"
    tier = "fast"
    system_prompt = (
        "You are a multimodal perception specialist. Your task is to extract structured "
        "information from visual inputs, documents, charts, tables, and audio transcripts. "
        "Always output clean, structured JSON or markdown tables. "
        "Be precise and exhaustive — extract ALL data points visible."
    )

    def __init__(self, client=None) -> None:
        super().__init__(client)
        self._selector = get_selector()

    async def perceive(
        self,
        ctx: AgentContext,
        content: str,
        content_type: str = "text",
        image_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract structured information from any content type."""
        task_description = f"Content type: {content_type}\n\n{content}"

        if image_urls:
            task_description += f"\n\nImage URLs: {', '.join(image_urls)}"

        spec, provider = self._selector.select(
            TaskType.PERCEPTION,
            require_multimodal=(content_type in ("image", "chart", "pdf")),
        )
        logger.info(f"[perception] model={spec.id} provider={provider} type={content_type}")

        messages = self._build_messages(ctx, task_description)
        messages[0]["content"] = self.system_prompt

        if provider == "nim":
            result = await self.client.chat(messages, model=spec.id, temperature=0.1, max_tokens=4096)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=spec.id, temperature=0.1, max_tokens=4096)

        structured = self.extract_json(result.get("content", ""))
        return {
            "raw": result.get("content", ""),
            "structured": structured,
            "model": spec.id,
            "provider": provider,
            "content_type": content_type,
        }

    async def parse_document(self, ctx: AgentContext, document_text: str) -> Dict[str, Any]:
        return await self.perceive(ctx, document_text, content_type="document")

    async def parse_chart(self, ctx: AgentContext, chart_description: str) -> Dict[str, Any]:
        return await self.perceive(ctx, chart_description, content_type="chart")

    async def transcribe_and_structure(self, ctx: AgentContext, transcript: str) -> Dict[str, Any]:
        return await self.perceive(ctx, transcript, content_type="transcript")
