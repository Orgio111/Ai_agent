"""Tool registry with failure-aware ranking and dynamic tool loading."""
from __future__ import annotations

import asyncio
import logging
import time

from .models import (
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolInfo,
    ToolRegisterRequest,
)
from .tools.code_tool import CodeExecutionTool
from .tools.filesystem_tool import FilesystemTool
from .tools.http_tool import HttpTool
from .tools.search_tool import SearchTool
from .tools.shell_tool import ShellTool

logger = logging.getLogger(__name__)


class ToolRecord:
    def __init__(self, info: ToolInfo, handler) -> None:
        self.info = info
        self.handler = handler
        self._success_count = 0
        self._failure_count = 0
        self._latencies: list[float] = []

    def record(self, success: bool, latency_ms: float) -> None:
        if success:
            self._success_count += 1
        else:
            self._failure_count += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies.pop(0)

        total = self._success_count + self._failure_count
        if total > 0:
            self.info.success_rate = self._success_count / total
        if self._latencies:
            self.info.avg_latency_ms = sum(self._latencies[-20:]) / len(self._latencies[-20:])

    @property
    def score(self) -> float:
        return self.info.success_rate / (self.info.avg_latency_ms + 1)


class ToolRegistry:
    def __init__(self, sandbox, permissions) -> None:
        self._tools: dict[str, ToolRecord] = {}
        self._sandbox = sandbox
        self._permissions = permissions

    async def load_builtin_tools(self) -> None:
        builtins = [
            ShellTool(sandbox=self._sandbox),
            FilesystemTool(sandbox=self._sandbox),
            HttpTool(),
            CodeExecutionTool(sandbox=self._sandbox),
            SearchTool(),
        ]
        for tool in builtins:
            record = ToolRecord(
                info=ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    category=tool.category,
                    parameters=tool.parameters,
                    required_role=tool.required_role,
                    sandboxed=tool.sandboxed,
                ),
                handler=tool,
            )
            self._tools[tool.name] = record
        logger.info(f"Loaded {len(self._tools)} builtin tools: {list(self._tools.keys())}")

    def register(self, req: ToolRegisterRequest) -> None:
        from .tools.remote_tool import RemoteTool
        tool = RemoteTool(
            name=req.name,
            description=req.description,
            category=req.category,
            endpoint=req.endpoint or "",
            parameters=req.parameters,
            required_role=req.required_role,
        )
        self._tools[req.name] = ToolRecord(
            info=ToolInfo(
                name=req.name,
                description=req.description,
                category=req.category,
                parameters=req.parameters,
                required_role=req.required_role,
                sandboxed=req.sandboxed,
            ),
            handler=tool,
        )
        logger.info(f"Registered tool: {req.name}")

    async def execute(self, req: ToolExecuteRequest) -> ToolExecuteResponse:
        record = self._tools.get(req.tool_name)
        if not record:
            raise KeyError(f"Tool not found: {req.tool_name}")

        start = time.monotonic()
        sandboxed = req.sandbox and record.info.sandboxed

        try:
            if sandboxed:
                result = await asyncio.wait_for(
                    self._sandbox.run(record.handler, req.args),
                    timeout=req.timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    record.handler.execute(req.args),
                    timeout=req.timeout_seconds,
                )

            latency = (time.monotonic() - start) * 1000
            record.record(success=True, latency_ms=latency)
            return ToolExecuteResponse(
                tool_name=req.tool_name,
                success=True,
                output=result,
                duration_ms=latency,
                sandboxed=sandboxed,
            )

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            record.record(success=False, latency_ms=latency)
            return ToolExecuteResponse(
                tool_name=req.tool_name,
                success=False,
                error=f"Timeout after {req.timeout_seconds}s",
                duration_ms=latency,
                sandboxed=sandboxed,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            record.record(success=False, latency_ms=latency)
            logger.error(f"Tool {req.tool_name} failed: {e}")
            return ToolExecuteResponse(
                tool_name=req.tool_name,
                success=False,
                error=str(e),
                duration_ms=latency,
                sandboxed=sandboxed,
            )

    def list_tools(self, role: str = "agent") -> list[ToolInfo]:
        return [
            r.info
            for r in sorted(self._tools.values(), key=lambda r: r.score, reverse=True)
            if self._permissions.can_execute(role, r.info.name)
        ]

    def stats(self) -> dict:
        return {
            name: {
                "success_rate": r.info.success_rate,
                "avg_latency_ms": r.info.avg_latency_ms,
                "score": r.score,
            }
            for name, r in self._tools.items()
        }
