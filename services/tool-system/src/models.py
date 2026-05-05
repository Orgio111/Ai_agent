from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ToolExecuteRequest(BaseModel):
    tool_name: str
    args: dict[str, Any] = {}
    session_id: Optional[str] = None
    timeout_seconds: float = 30.0
    sandbox: bool = True


class ToolExecuteResponse(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    sandboxed: bool = False


class ToolInfo(BaseModel):
    name: str
    description: str
    category: str
    parameters: dict[str, Any] = {}
    required_role: str = "agent"
    sandboxed: bool = True
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]


class ToolRegisterRequest(BaseModel):
    name: str
    description: str
    category: str
    endpoint: Optional[str] = None
    parameters: dict[str, Any] = {}
    required_role: str = "agent"
    sandboxed: bool = True
