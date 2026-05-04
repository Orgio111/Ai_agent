"""Tool abstraction shared by every tool implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "output": self.output, "error": self.error, "meta": self.meta}


class Tool(ABC):
    name: str = "tool"
    description: str = ""
    schema: Dict[str, Any] = {}

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "schema": self.schema}
