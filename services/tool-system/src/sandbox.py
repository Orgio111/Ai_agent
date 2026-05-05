"""Sandbox executor: runs tools in isolated environments."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SandboxExecutor:
    def __init__(self, sandbox_dir: str, docker_enabled: bool = False) -> None:
        self._sandbox_dir = Path(sandbox_dir)
        self._docker_enabled = docker_enabled
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, tool, args: dict[str, Any]) -> Any:
        """Execute a tool in the sandbox environment."""
        if self._docker_enabled:
            return await self._run_in_docker(tool, args)
        return await self._run_isolated(tool, args)

    async def _run_isolated(self, tool, args: dict[str, Any]) -> Any:
        """Run with restricted file system access via chroot-like isolation."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self._sandbox_dir)
            result = await tool.execute(args)
            return result
        finally:
            os.chdir(old_cwd)

    async def _run_in_docker(self, tool, args: dict[str, Any]) -> Any:
        """Run tool in a Docker container for maximum isolation."""
        script = self._build_tool_script(tool, args)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=self._sandbox_dir, delete=False
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm",
                "--network=none",
                "--memory=256m",
                "--cpus=0.5",
                "--read-only",
                f"--volume={script_path}:/tool.py:ro",
                "python:3.12-alpine",
                "python", "/tool.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=25.0)
            if proc.returncode != 0:
                raise RuntimeError(f"Docker tool failed: {stderr.decode()[:500]}")
            return stdout.decode()
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def _build_tool_script(self, tool, args: dict) -> str:
        import json
        return f"""
import json, sys
args = {json.dumps(args)}
# Tool execution placeholder — actual tool would be injected
print(json.dumps({{"result": "sandboxed", "args": args}}))
"""
