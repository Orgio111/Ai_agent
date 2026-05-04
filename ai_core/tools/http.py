"""HTTP tool for external API calls."""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ..config import get_settings
from .base import Tool, ToolResult


class HttpTool(Tool):
    name = "http"
    description = "Perform an HTTP request (GET/POST/PUT/DELETE) and return the response."
    schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
            "url": {"type": "string"},
            "headers": {"type": "object"},
            "params": {"type": "object"},
            "json_body": {"type": "object"},
            "data": {"type": "string"},
        },
        "required": ["method", "url"],
    }

    def __init__(self) -> None:
        s = get_settings()
        cfg = s.raw_settings.get("tools", {}).get("http", {})
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.timeout: float = float(cfg.get("timeout_seconds", 20))

    async def run(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        if not self.enabled:
            return ToolResult(ok=False, error="http tool disabled")
        method = method.upper()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(
                    method, url,
                    headers=headers or {},
                    params=params or None,
                    json=json_body,
                    content=data.encode("utf-8") if data else None,
                )
        except httpx.HTTPError as e:
            return ToolResult(ok=False, error=f"http error: {e}")

        body: Any
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
        else:
            body = resp.text[:8000]

        return ToolResult(
            ok=resp.is_success,
            output={"status": resp.status_code, "body": body, "headers": dict(resp.headers)},
            error="" if resp.is_success else f"status {resp.status_code}",
        )
