"""Web search tool using DuckDuckGo (no API key required)."""
from __future__ import annotations

from typing import Any
import httpx


class SearchTool:
    name = "search"
    description = "Search the web using DuckDuckGo"
    category = "search"
    required_role = "agent"
    sandboxed = False
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query", "").strip()
        max_results = int(args.get("max_results", 5))

        if not query:
            return {"error": "No query provided"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1",
                    },
                    headers={"User-Agent": "JARVIS/1.0 (research agent)"},
                )
                if resp.status_code >= 400:
                    return {"error": f"Search API returned {resp.status_code}"}

                data = resp.json()
                results = []

                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", ""),
                        "snippet": data["AbstractText"][:500],
                        "url": data.get("AbstractURL", ""),
                    })

                for item in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(item, dict) and "Text" in item:
                        results.append({
                            "title": item.get("Text", "")[:100],
                            "snippet": item.get("Text", "")[:300],
                            "url": item.get("FirstURL", ""),
                        })

                return {"query": query, "results": results[:max_results]}
        except Exception as e:
            return {"error": str(e), "query": query}


class RemoteTool:
    def __init__(
        self, name: str, description: str, category: str,
        endpoint: str, parameters: dict, required_role: str = "agent"
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.endpoint = endpoint
        self.parameters = parameters
        self.required_role = required_role
        self.sandboxed = False

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.endpoint, json=args)
            resp.raise_for_status()
            return resp.json()
