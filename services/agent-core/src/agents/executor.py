"""SmartExecutor Agent: dynamic tool selection, self-correction, and plan repair."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import httpx

from .base import AgentBase


class SmartExecutorAgent(AgentBase):
    name = "executor"
    tier = "balanced"
    system_prompt = """You are an expert Executor AI agent in a JARVIS-class autonomous system.

Your role:
1. Execute assigned tasks using available tools
2. Select the most appropriate tool for each step
3. Handle errors gracefully with self-correction
4. Return structured results

When using tools, format calls as:
<tool_call>{"tool": "tool_name", "args": {"key": "value"}}</tool_call>

Available tool categories: shell, filesystem, http, code_execution, search, database

After tool execution, synthesize results into a coherent response.
If a tool fails, try an alternative approach — do NOT simply retry the same failed call.
Document what worked and what didn't for future learning."""

    def __init__(self, *args, max_retries: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries
        self._tool_success_rates: dict[str, float] = {}

    async def execute(
        self,
        task: dict[str, Any],
        session_id: str,
        history: list[dict],
        context: dict[str, Any],
        available_tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        task_desc = task.get("description", "")
        tool_history: list[dict] = []
        result_content = ""

        for attempt in range(self._max_retries):
            result = await self.run(
                prompt=self._build_task_prompt(task_desc, tool_history, attempt),
                session_id=session_id,
                history=history,
                context=context,
                tools=available_tools,
            )

            content = result["content"]
            tool_calls = result.get("tool_calls", [])

            if not tool_calls:
                result_content = content
                break

            # Execute tools in parallel where dependencies allow
            tool_results = await self._execute_tools_parallel(tool_calls, session_id)
            tool_history.append({"calls": tool_calls, "results": tool_results})

            # Check if all tools succeeded
            all_success = all(not r.get("error") for r in tool_results)
            if all_success:
                result_content = self._synthesize_results(content, tool_results)
                break

            # Self-correction: inform agent of failures
            failed = [r for r in tool_results if r.get("error")]
            context["tool_failures"] = failed

        return {
            "content": result_content or content,
            "tool_calls": tool_history,
            "attempts": attempt + 1,
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
        }

    async def _execute_tools_parallel(
        self, tool_calls: list[dict], session_id: str
    ) -> list[dict]:
        tasks = [
            self.execute_tool(call.get("tool", ""), call.get("args", {}))
            for call in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({"error": str(r), "tool": tool_calls[i].get("tool")})
                self._update_tool_rate(tool_calls[i].get("tool", ""), success=False)
            else:
                processed.append(r)
                self._update_tool_rate(tool_calls[i].get("tool", ""), success=True)
        return processed

    def _build_task_prompt(
        self, task: str, history: list[dict], attempt: int
    ) -> str:
        if attempt == 0:
            return f"Execute the following task:\n{task}"

        failures = "\n".join(
            f"- {h['calls'][i].get('tool')}: {h['results'][i].get('error', 'unknown')}"
            for h in history
            for i, r in enumerate(h["results"])
            if r.get("error")
        )
        return (
            f"Execute the following task (attempt {attempt + 1}/{self._max_retries}):\n{task}\n\n"
            f"Previous failures to avoid:\n{failures}\n\n"
            "Use a different approach to accomplish the task."
        )

    def _synthesize_results(self, content: str, tool_results: list[dict]) -> str:
        successful = [r for r in tool_results if not r.get("error")]
        if not successful:
            return content
        summary = json.dumps(
            [{"tool": r.get("tool_name"), "output": str(r.get("output", ""))[:300]}
             for r in successful],
            indent=2,
        )
        return f"{content}\n\nTool Results:\n{summary}"

    def _update_tool_rate(self, tool: str, success: bool) -> None:
        current = self._tool_success_rates.get(tool, 1.0)
        if success:
            self._tool_success_rates[tool] = min(1.0, current + 0.02)
        else:
            self._tool_success_rates[tool] = max(0.0, current - 0.1)
