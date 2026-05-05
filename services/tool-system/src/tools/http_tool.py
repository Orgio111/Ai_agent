"""HTTP tool for fetching URLs and making API calls."""
from __future__ import annotations

import re
from typing import Any

import httpx

BLOCKED_PATTERNS = re.compile(
    r"(localhost|127\.0\.0\.1|0\.0\.0\.0|169\.254\.|192\.168\.|10\.\d+\.\d+\.|172\.(1[6-9]|2\d|3[01])\.)",
    re.IGNORECASE,
)


class HttpTool:
    name = "http_get"
    description = "Fetch a URL via HTTP GET"
    category = "http"
    required_role = "agent"
    sandboxed = False
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "headers": {"type": "object"},
            "timeout": {"type": "number", "default": 15},
        },
        "required": ["url"],
    }

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        url = args.get("url", "").strip()
        if not url:
            return {"error": "No URL provided"}

        if BLOCKED_PATTERNS.search(url):
            return {"error": "SSRF blocked: internal network addresses not allowed"}

        if not url.startswith(("http://", "https://")):
            return {"error": "Only http/https URLs allowed"}

        headers = args.get("headers", {})
        timeout = float(args.get("timeout", 15))

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
            ) as client:
                resp = await client.get(url, headers=headers)
                content = resp.text[:50000]
                return {
                    "url": str(resp.url),
                    "status_code": resp.status_code,
                    "content": content,
                    "content_type": resp.headers.get("content-type", ""),
                    "content_length": len(content),
                }
        except httpx.TimeoutException:
            return {"error": f"Request timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}
