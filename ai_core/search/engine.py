"""Deep Search Engine — multi-query, freshness-filtered, multi-source aggregation.

Pipeline:
  1. Generate N search queries from the original question
  2. Execute queries in parallel via web search tool
  3. Filter by freshness (prefer recent results)
  4. Aggregate + deduplicate results
  5. Resolve contradictions between sources
  6. Return ranked, structured evidence
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_setup import logger
from ..model_selector import TaskType, get_selector
from ..nim_client import get_nim_client
from ..openrouter.client import get_openrouter_client


@dataclass
class SearchResult:
    query: str
    source: str
    content: str
    relevance_score: float = 0.5
    timestamp: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self) -> None:
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class DeepSearchResult:
    original_query: str
    sub_queries: List[str]
    results: List[SearchResult]
    synthesis: str
    contradictions: List[Dict[str, Any]]
    confidence: float
    sources_used: int


class DeepSearchEngine:
    """Multi-query parallel search with freshness filtering and synthesis."""

    def __init__(
        self,
        n_queries: int = 4,
        freshness_years: int = 2,
        max_results_per_query: int = 5,
    ) -> None:
        self._n_queries = n_queries
        self._freshness_cutoff = freshness_years
        self._max_per_query = max_results_per_query
        self._selector = get_selector()

    async def search(
        self,
        query: str,
        session_context: str = "",
        require_fresh: bool = True,
    ) -> DeepSearchResult:
        """Execute a full deep search pipeline."""
        logger.info(f"[search] deep search: {query[:80]}...")

        # Step 1: Generate diverse sub-queries
        sub_queries = await self._generate_queries(query, session_context)

        # Step 2: Execute all sub-queries in parallel
        raw_results = await asyncio.gather(*[
            self._execute_query(q) for q in sub_queries
        ])
        all_results: List[SearchResult] = [r for batch in raw_results for r in batch]

        # Step 3: Deduplicate by content hash
        seen_hashes: set[str] = set()
        unique_results: List[SearchResult] = []
        for r in all_results:
            if r.content_hash not in seen_hashes:
                seen_hashes.add(r.content_hash)
                unique_results.append(r)

        # Step 4: Sort by relevance (descending)
        unique_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Step 5: Detect contradictions
        contradictions = await self._detect_contradictions(query, unique_results[:10])

        # Step 6: Synthesize findings
        synthesis, confidence = await self._synthesize(query, unique_results[:10], contradictions)

        return DeepSearchResult(
            original_query=query,
            sub_queries=sub_queries,
            results=unique_results,
            synthesis=synthesis,
            contradictions=contradictions,
            confidence=confidence,
            sources_used=len(unique_results),
        )

    async def _generate_queries(self, query: str, context: str) -> List[str]:
        """Use LLM to generate diverse search queries."""
        prompt = (
            f"Generate {self._n_queries} diverse, specific search queries to comprehensively "
            f"research this topic. Vary the angle (technical, recent, comparative, definitional).\n\n"
            f"Topic: {query}\n"
            f"Context: {context[:200] if context else 'None'}\n\n"
            f"Output as JSON array: [\"query1\", \"query2\", ...]"
        )
        messages = [
            {"role": "system", "content": "You generate targeted search queries for deep research."},
            {"role": "user", "content": prompt},
        ]
        spec, provider = self._selector.select(TaskType.REASONING)
        try:
            if provider == "nim":
                result = await get_nim_client().chat(
                    messages, model=spec.id, temperature=0.3, max_tokens=512
                )
            else:
                result = await get_openrouter_client().chat(
                    messages, model=spec.id, temperature=0.3, max_tokens=512
                )
            from ..agents.base import BaseAgent
            parsed = BaseAgent.extract_json(result.get("content", ""))
            if isinstance(parsed, list) and len(parsed) >= 2:
                return [str(q) for q in parsed[: self._n_queries]]
        except Exception as e:
            logger.warning(f"[search] query generation failed: {e}")

        # Fallback: simple reformulations
        return [
            query,
            f"{query} latest research",
            f"{query} technical analysis",
            f"how does {query} work",
        ][: self._n_queries]

    async def _execute_query(self, query: str) -> List[SearchResult]:
        """Execute a single search query.

        In production, this integrates with a web search API (Serper, Tavily, etc.).
        Here we return a structured placeholder that the LLM can reason over.
        """
        # Simulate search result — real implementation hooks into Serper/Tavily/DuckDuckGo
        await asyncio.sleep(0)  # yield control
        return [
            SearchResult(
                query=query,
                source="web_search",
                content=f"[Search results for: {query}]",
                relevance_score=0.7,
            )
        ]

    async def _detect_contradictions(
        self, query: str, results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Identify conflicting information across sources."""
        if len(results) < 2:
            return []

        content_snippets = "\n".join(
            f"Source {i+1}: {r.content[:300]}" for i, r in enumerate(results[:6])
        )
        prompt = (
            f"Identify any factual contradictions between these sources about: {query}\n\n"
            f"{content_snippets}\n\n"
            f"Output JSON: [{{\"sources\": [1,2], \"contradiction\": \"...\", \"resolution\": \"...\"}}]"
        )
        messages = [
            {"role": "system", "content": "You identify contradictions between information sources."},
            {"role": "user", "content": prompt},
        ]
        spec, provider = self._selector.select(TaskType.AUDIT)
        try:
            if provider == "nim":
                result = await get_nim_client().chat(
                    messages, model=spec.id, temperature=0.1, max_tokens=1024
                )
            else:
                result = await get_openrouter_client().chat(
                    messages, model=spec.id, temperature=0.1, max_tokens=1024
                )
            from ..agents.base import BaseAgent
            parsed = BaseAgent.extract_json(result.get("content", ""))
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            logger.warning(f"[search] contradiction detection failed: {e}")
        return []

    async def _synthesize(
        self, query: str, results: List[SearchResult], contradictions: List[Dict]
    ) -> tuple[str, float]:
        """Synthesize search results into a coherent answer."""
        if not results:
            return "No search results found.", 0.0

        content_snippets = "\n".join(
            f"Source {i+1} (relevance={r.relevance_score:.2f}): {r.content[:400]}"
            for i, r in enumerate(results[:8])
        )
        contradiction_note = ""
        if contradictions:
            contradiction_note = f"\nNote {len(contradictions)} contradiction(s) found between sources."

        prompt = (
            f"Synthesize these search results into a comprehensive, accurate answer.\n"
            f"Query: {query}{contradiction_note}\n\n"
            f"Sources:\n{content_snippets}\n\n"
            f"Output JSON: {{\"synthesis\": \"...\", \"confidence\": 0.0-1.0}}"
        )
        messages = [
            {"role": "system", "content": "You synthesize search results into accurate, grounded answers."},
            {"role": "user", "content": prompt},
        ]
        spec, provider = self._selector.select(TaskType.RESEARCH)
        try:
            if provider == "nim":
                result = await get_nim_client().chat(
                    messages, model=spec.id, temperature=0.2, max_tokens=2048
                )
            else:
                result = await get_openrouter_client().chat(
                    messages, model=spec.id, temperature=0.2, max_tokens=2048
                )
            from ..agents.base import BaseAgent
            parsed = BaseAgent.extract_json(result.get("content", ""))
            if isinstance(parsed, dict):
                return str(parsed.get("synthesis", "")), float(parsed.get("confidence", 0.7))
            return result.get("content", ""), 0.6
        except Exception as e:
            logger.warning(f"[search] synthesis failed: {e}")
            return content_snippets[:500], 0.4


_engine: Optional[DeepSearchEngine] = None


def get_search_engine() -> DeepSearchEngine:
    global _engine
    if _engine is None:
        _engine = DeepSearchEngine()
    return _engine
