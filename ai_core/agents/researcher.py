"""Research Agent — DeepSeek R1 (primary), reasoning swarm (support).

Handles: deep multi-query search, macro/micro analysis, hypothesis generation,
strategy creation, evidence aggregation.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from ..logging_setup import logger
from ..model_selector import TaskType, get_selector
from ..openrouter.client import get_openrouter_client
from .base import AgentContext, BaseAgent

_RESEARCH_SYSTEM = """\
You are an elite research analyst with deep expertise in financial markets, technology, \
and scientific domains. You think step-by-step using chain-of-thought reasoning.

When given a research query:
1. Decompose it into 3-5 sub-questions
2. Reason through each sub-question systematically
3. Synthesize findings into a coherent analysis
4. Identify knowledge gaps and confidence levels
5. Output a structured research report in JSON:
   {
     "summary": "...",
     "sub_analyses": [...],
     "hypotheses": [...],
     "confidence": 0.0-1.0,
     "gaps": [...],
     "strategy_recommendation": "..."
   }
"""


class ResearchAgent(BaseAgent):
    name = "researcher"
    tier = "complex"
    system_prompt = _RESEARCH_SYSTEM

    def __init__(self, client=None) -> None:
        super().__init__(client)
        self._selector = get_selector()

    async def research(
        self,
        ctx: AgentContext,
        query: str,
        depth: int = 3,
        parallel_queries: bool = True,
    ) -> Dict[str, Any]:
        """Run deep research on a query using chain-of-thought reasoning."""
        spec, provider = self._selector.select(TaskType.RESEARCH)
        logger.info(f"[researcher] model={spec.id} provider={provider} depth={depth}")

        # Generate sub-queries for parallel research
        sub_queries = await self._generate_sub_queries(ctx, query, n=depth)

        if parallel_queries and len(sub_queries) > 1:
            sub_results = await asyncio.gather(*[
                self._research_sub_query(ctx, q, spec.id, provider)
                for q in sub_queries
            ])
        else:
            sub_results = []
            for q in sub_queries:
                sub_results.append(await self._research_sub_query(ctx, q, spec.id, provider))

        synthesis_prompt = (
            f"Original query: {query}\n\n"
            f"Sub-analyses:\n" +
            "\n".join(f"{i+1}. {r.get('analysis', '')}" for i, r in enumerate(sub_results)) +
            "\n\nSynthesize these into a final structured research report."
        )

        messages = self._build_messages(ctx, synthesis_prompt)
        if provider == "nim":
            result = await self.client.chat(messages, model=spec.id, temperature=0.3, max_tokens=4096)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=spec.id, temperature=0.3, max_tokens=4096)

        structured = self.extract_json(result.get("content", "")) or {}
        return {
            "query": query,
            "sub_results": sub_results,
            "synthesis": result.get("content", ""),
            "structured": structured,
            "model": spec.id,
            "provider": provider,
        }

    async def _generate_sub_queries(
        self, ctx: AgentContext, query: str, n: int
    ) -> List[str]:
        """Use the model to decompose a complex query into sub-questions."""
        prompt = (
            f"Break down this research query into exactly {n} specific, searchable sub-questions:\n\n"
            f"Query: {query}\n\n"
            f"Output as a JSON array of strings: [\"sub-q1\", \"sub-q2\", ...]"
        )
        messages = [
            {"role": "system", "content": "You decompose complex research queries into sub-questions."},
            {"role": "user", "content": prompt},
        ]
        spec, provider = self._selector.select(TaskType.REASONING)
        if provider == "nim":
            result = await self.client.chat(messages, model=spec.id, temperature=0.2, max_tokens=512)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=spec.id, temperature=0.2, max_tokens=512)

        parsed = self.extract_json(result.get("content", ""))
        if isinstance(parsed, list):
            return [str(q) for q in parsed[:n]]
        return [query]

    async def _research_sub_query(
        self, ctx: AgentContext, query: str, model_id: str, provider: str
    ) -> Dict[str, Any]:
        messages = self._build_messages(ctx, f"Research this specific question: {query}")
        if provider == "nim":
            result = await self.client.chat(messages, model=model_id, temperature=0.4, max_tokens=2048)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=model_id, temperature=0.4, max_tokens=2048)
        return {"query": query, "analysis": result.get("content", "")}
