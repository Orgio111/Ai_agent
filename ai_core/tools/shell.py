"""Sandboxed shell command tool.

Only commands whose first token is in the configured allow-list run.
Execution uses argv (no shell=True) so command injection via arguments
is contained.
"""
from __future__ import annotations

import asyncio
import shlex
from typing import Any, List

from ..config import get_settings
from .base import Tool, ToolResult


class ShellTool(Tool):
    name = "shell"
    description = "Run a whitelisted shell command and return stdout/stderr."
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command line to execute"},
            "timeout": {"type": "number", "default": 15},
        },
        "required": ["command"],
    }

    def __init__(self) -> None:
        s = get_settings()
        tools_cfg = s.raw_settings.get("tools", {}).get("shell", {})
        self.enabled: bool = bool(tools_cfg.get("enabled", True))
        self.allowed: List[str] = list(tools_cfg.get("allowed_commands", []))

    async def run(self, command: str, timeout: float = 15.0, **_: Any) -> ToolResult:
        if not self.enabled:
            return ToolResult(ok=False, error="shell tool disabled")

        try:
            argv = shlex.split(command)
        except ValueError as e:
            return ToolResult(ok=False, error=f"invalid command: {e}")

        if not argv:
            return ToolResult(ok=False, error="empty command")

        head = argv[0]
        if self.allowed and head not in self.allowed:
            return ToolResult(
                ok=False,
                error=f"command '{head}' not in allow-list: {self.allowed}",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, error=f"timeout after {timeout}s")
        except FileNotFoundError:
            return ToolResult(ok=False, error=f"command not found: {head}")

        return ToolResult(
            ok=proc.returncode == 0,
            output={
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "returncode": proc.returncode,
            },
            error="" if proc.returncode == 0 else f"exit {proc.returncode}",
            meta={"argv": argv},
        )
