"""Research Agent: gathers information, synthesizes knowledge, resolves unknowns."""
from __future__ import annotations

from typing import Any

from .base import AgentBase


class ResearchAgent(AgentBase):
    name = "researcher"
    tier = "complex"
    system_prompt = """You are an expert Research AI agent in a JARVIS-class autonomous system.

Your role:
1. Gather comprehensive information about the topic
2. Synthesize findings from multiple sources/perspectives
3. Identify knowledge gaps and uncertainties
4. Provide evidence-based conclusions with confidence levels
5. Structure findings for consumption by other agents

Use tools like http_get to fetch web resources when needed.
Always cite confidence levels: HIGH (>90%), MEDIUM (60-90%), LOW (<60%).

Format as structured Markdown with sections:
## Summary
## Key Findings
## Evidence
## Uncertainties
## Recommendations for other agents"""

    async def research(
        self,
        topic: str,
        session_id: str,
        history: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "http_get",
                    "description": "Fetch a URL to gather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                },
            }
        ]

        result = await self.run(
            prompt=f"Research the following topic thoroughly:\n{topic}",
            session_id=session_id,
            history=history,
            context=context,
            tools=tools,
        )

        tool_calls = result.get("tool_calls", [])
        fetch_results: list[dict] = []
        for call in tool_calls:
            if call.get("tool") == "http_get":
                fetch_result = await self.execute_tool("http_get", call.get("args", {}))
                fetch_results.append(fetch_result)

        final_content = result["content"]
        if fetch_results:
            sources = "\n".join(
                f"- {r.get('url', '')}: {str(r.get('content', ''))[:300]}"
                for r in fetch_results
                if not r.get("error")
            )
            if sources:
                final_content += f"\n\nSources consulted:\n{sources}"

        # Store research findings as semantic memory
        await self._store_memory(
            f"Research findings on '{topic}': {final_content[:500]}",
            session_id,
            memory_type="semantic",
        )

        return {
            "content": final_content,
            "sources": fetch_results,
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
        }
