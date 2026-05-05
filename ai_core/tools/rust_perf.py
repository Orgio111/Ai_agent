"""Bridge to the Rust performance microservice.

The Rust crate exposes performance-critical primitives (vector ops,
checksums, parallel similarity).  This tool calls them over HTTP so the
Python core stays language-agnostic.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ..config import get_settings
from .base import Tool, ToolResult


class RustPerfTool(Tool):
    name = "rust_perf"
    description = (
        "Invoke the Rust performance service. Operations: cosine_batch, "
        "checksum, normalize."
    )
    schema = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["cosine_batch", "checksum", "normalize", "health"]},
            "payload": {"type": "object"},
        },
        "required": ["op"],
    }

    def __init__(self) -> None:
        s = get_settings()
        cfg = s.raw_settings.get("tools", {}).get("rust_perf", {})
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.url: str = s.rust_perf_url.rstrip("/")

    async def run(self, op: str, payload: Optional[Dict[str, Any]] = None, **_: Any) -> ToolResult:
        if not self.enabled:
            return ToolResult(ok=False, error="rust_perf tool disabled")
        path_map = {
            "cosine_batch": "/cosine_batch",
            "checksum": "/checksum",
            "normalize": "/normalize",
            "health": "/health",
        }
        if op not in path_map:
            return ToolResult(ok=False, error=f"unknown rust_perf op '{op}'")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if op == "health":
                    resp = await client.get(f"{self.url}{path_map[op]}")
                else:
                    resp = await client.post(f"{self.url}{path_map[op]}", json=payload or {})
        except httpx.HTTPError as e:
            return ToolResult(ok=False, error=f"rust service unreachable: {e}")

        if not resp.is_success:
            return ToolResult(ok=False, error=f"status {resp.status_code}", output=resp.text)
        try:
            return ToolResult(ok=True, output=resp.json())
        except ValueError:
            return ToolResult(ok=True, output=resp.text)
