"""Filesystem tool with sandbox path enforcement."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class FilesystemTool:
    name = "filesystem"
    description = "Read and write files within the sandbox"
    category = "filesystem"
    required_role = "operator"
    sandboxed = True
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["read", "write", "list", "exists"]},
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["operation", "path"],
    }

    def __init__(self, sandbox) -> None:
        self._root = sandbox._sandbox_dir

    def _safe_path(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise PermissionError(f"Path escape detected: {path}")
        return resolved

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        op = args.get("operation", "read")
        path_str = args.get("path", "")

        try:
            safe = self._safe_path(path_str)
        except PermissionError as e:
            return {"error": str(e)}

        if op == "read":
            if not safe.exists():
                return {"error": f"File not found: {path_str}"}
            if not safe.is_file():
                return {"error": f"Not a file: {path_str}"}
            content = safe.read_text(errors="replace")[:100000]
            return {"content": content, "size": safe.stat().st_size}

        elif op == "write":
            content = args.get("content", "")
            safe.parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content)
            return {"written": len(content), "path": str(safe)}

        elif op == "list":
            if not safe.exists():
                return {"error": f"Directory not found: {path_str}"}
            entries = [
                {"name": e.name, "type": "dir" if e.is_dir() else "file", "size": e.stat().st_size if e.is_file() else 0}
                for e in sorted(safe.iterdir())[:100]
            ]
            return {"entries": entries, "count": len(entries)}

        elif op == "exists":
            return {"exists": safe.exists(), "is_file": safe.is_file(), "is_dir": safe.is_dir()}

        return {"error": f"Unknown operation: {op}"}


class FilesystemReadTool(FilesystemTool):
    name = "filesystem_read"
    description = "Read files within the sandbox (read-only)"
    required_role = "agent"

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("operation") == "write":
            return {"error": "Write not permitted for filesystem_read tool"}
        return await super().execute(args)
