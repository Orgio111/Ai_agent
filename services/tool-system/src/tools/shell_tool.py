"""Shell command execution tool with allowlist enforcement."""
from __future__ import annotations

import asyncio
import shlex
from typing import Any

ALLOWED_COMMANDS = frozenset([
    "ls", "echo", "cat", "pwd", "uname", "whoami", "date", "df", "du",
    "ps", "head", "tail", "grep", "find", "wc", "sort", "uniq", "cut",
    "sed", "awk", "tr", "diff", "md5sum", "sha256sum", "curl", "wget",
    "python3", "pip", "node", "npm", "git", "make", "cmake", "gcc",
])


class ShellTool:
    name = "shell"
    description = "Execute shell commands in a restricted environment"
    category = "shell"
    required_role = "operator"
    sandboxed = True
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "default": 10},
        },
        "required": ["command"],
    }

    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command", "").strip()
        timeout = float(args.get("timeout", 10))

        if not command:
            return {"error": "No command provided"}

        # Validate command against allowlist
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return {"error": f"Invalid command syntax: {e}"}

        base_cmd = tokens[0].split("/")[-1] if tokens else ""
        if base_cmd not in ALLOWED_COMMANDS:
            return {"error": f"Command '{base_cmd}' not in allowlist"}

        # Reject dangerous patterns
        dangerous = ["&&", "||", ";", "|", ">", "<", "`", "$(",
                     "rm", "rmdir", "mv", "chmod", "chown", "kill", "pkill"]
        if any(d in command for d in dangerous):
            return {"error": "Dangerous pattern detected in command"}

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
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
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}
