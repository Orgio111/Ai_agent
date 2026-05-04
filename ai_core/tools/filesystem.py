"""Sandboxed filesystem tool.

All operations resolve paths and confirm they live inside the configured
sandbox directory. Anything outside is rejected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from ..config import get_settings
from .base import Tool, ToolResult


class FilesystemTool(Tool):
    name = "filesystem"
    description = (
        "Read/write/list files inside a sandbox directory. "
        "Operations: read, write, append, list, delete, exists."
    )
    schema = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["read", "write", "append", "list", "delete", "exists"],
            },
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["op", "path"],
    }

    def __init__(self) -> None:
        s = get_settings()
        cfg = s.raw_settings.get("tools", {}).get("filesystem", {})
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.sandbox: Path = Path(cfg.get("sandbox_dir", "./data/sandbox")).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        p = (self.sandbox / path).resolve()
        try:
            p.relative_to(self.sandbox)
        except ValueError as e:
            raise PermissionError(f"path '{path}' escapes sandbox") from e
        return p

    async def run(self, op: str, path: str, content: str = "", **_: Any) -> ToolResult:
        if not self.enabled:
            return ToolResult(ok=False, error="filesystem tool disabled")
        try:
            target = self._resolve(path)
        except PermissionError as e:
            return ToolResult(ok=False, error=str(e))

        try:
            if op == "read":
                if not target.exists():
                    return ToolResult(ok=False, error="not found")
                return ToolResult(ok=True, output=target.read_text(encoding="utf-8"))
            if op == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return ToolResult(ok=True, output=f"wrote {len(content)} bytes")
            if op == "append":
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(content)
                return ToolResult(ok=True, output=f"appended {len(content)} bytes")
            if op == "list":
                base = target if target.is_dir() else target.parent
                entries: List[str] = sorted(p.name for p in base.iterdir())
                return ToolResult(ok=True, output=entries)
            if op == "delete":
                if target.exists():
                    if target.is_dir():
                        return ToolResult(ok=False, error="refusing to delete directory")
                    target.unlink()
                    return ToolResult(ok=True, output="deleted")
                return ToolResult(ok=False, error="not found")
            if op == "exists":
                return ToolResult(ok=True, output=target.exists())
        except OSError as e:
            return ToolResult(ok=False, error=f"OS error: {e}")

        return ToolResult(ok=False, error=f"unknown op '{op}'")
