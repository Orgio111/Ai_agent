"""Code execution tool: runs Python/JS snippets in isolated subprocess."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

BLOCKED_IMPORTS = frozenset([
    "os.system", "subprocess", "socket", "requests", "urllib",
    "open(", "__import__", "exec(", "eval(", "compile(",
    "importlib", "ctypes", "multiprocessing",
])


class CodeExecutionTool:
    name = "code_execution"
    description = "Execute Python code snippets in a restricted sandbox"
    category = "code"
    required_role = "agent"
    sandboxed = True
    parameters = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "enum": ["python"], "default": "python"},
            "code": {"type": "string"},
            "timeout": {"type": "number", "default": 10},
        },
        "required": ["code"],
    }

    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        code = args.get("code", "")
        language = args.get("language", "python")
        timeout = float(args.get("timeout", 10))

        if not code.strip():
            return {"error": "No code provided"}

        # Static analysis: block dangerous patterns
        for blocked in BLOCKED_IMPORTS:
            if blocked in code:
                return {"error": f"Blocked pattern: {blocked}"}

        if language != "python":
            return {"error": f"Language '{language}' not supported"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py",
            dir=self._sandbox._sandbox_dir,
            delete=False,
        ) as f:
            f.write(self._wrap_code(code))
            script = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._sandbox._sandbox_dir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode(errors="replace")[:10000],
                "stderr": stderr.decode(errors="replace")[:2000],
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"error": f"Code timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass

    def _wrap_code(self, code: str) -> str:
        return f"""
import sys
import math
import json
import re
import datetime
import collections
import itertools
import functools
import random
import string

# User code:
{code}
"""
