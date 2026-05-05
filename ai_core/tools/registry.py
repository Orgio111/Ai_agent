"""Tool registry - central catalogue of available tools."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..logging_setup import logger
from .base import Tool, ToolResult
from .filesystem import FilesystemTool
from .http import HttpGetTool, HttpTool
from .rust_perf import RustPerfTool
from .shell import ShellTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for tool in (ShellTool(), FilesystemTool(), HttpGetTool(), HttpTool(), RustPerfTool()):
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        """Return Tool objects directly so callers can access .name, .description, etc."""
        return list(self._tools.values())

    async def run(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool '{name}'")
        try:
            return await tool.run(**kwargs)
        except TypeError as e:
            return ToolResult(ok=False, error=f"bad arguments: {e}")
        except Exception as e:  # pragma: no cover
            logger.exception(f"tool '{name}' crashed")
            return ToolResult(ok=False, error=f"tool crash: {e}")


_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
