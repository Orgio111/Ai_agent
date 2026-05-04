"""Smart Tool System: tool registry, permission enforcement, sandbox execution,
failure-aware ranking, and plan repair."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .registry import ToolRegistry
from .sandbox import SandboxExecutor
from .permissions import PermissionManager
from .models import (
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolListResponse,
    ToolRegisterRequest,
)

registry: ToolRegistry | None = None
sandbox: SandboxExecutor | None = None
permissions: PermissionManager | None = None
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global registry, sandbox, permissions

    permissions = PermissionManager(
        secret_key=os.environ.get("TOOL_AUTH_SECRET", "jarvis-dev-secret")
    )
    sandbox = SandboxExecutor(
        sandbox_dir=os.environ.get("SANDBOX_DIR", "/tmp/jarvis-sandbox"),
        docker_enabled=os.environ.get("DOCKER_SANDBOX", "false").lower() == "true",
    )
    registry = ToolRegistry(sandbox=sandbox, permissions=permissions)
    await registry.load_builtin_tools()

    yield


app = FastAPI(title="JARVIS Tool System", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/execute", response_model=ToolExecuteResponse)
async def execute(
    req: ToolExecuteRequest,
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> ToolExecuteResponse:
    assert registry is not None and permissions is not None

    role = "agent"
    if credentials:
        role = permissions.verify_token(credentials.credentials)

    if not permissions.can_execute(role, req.tool_name):
        raise HTTPException(403, f"Role '{role}' cannot execute tool '{req.tool_name}'")

    try:
        return await registry.execute(req)
    except KeyError:
        raise HTTPException(404, f"Tool '{req.tool_name}' not found")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/tools", response_model=ToolListResponse)
async def list_tools(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> ToolListResponse:
    assert registry is not None and permissions is not None
    role = permissions.verify_token(credentials.credentials if credentials else "") if credentials else "agent"
    return ToolListResponse(tools=registry.list_tools(role=role))


@app.post("/register")
async def register_tool(req: ToolRegisterRequest) -> dict:
    assert registry is not None
    registry.register(req)
    return {"tool_name": req.name, "status": "registered"}


@app.get("/stats")
async def tool_stats() -> dict:
    assert registry is not None
    return registry.stats()


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "tools_loaded": len(registry.list_tools()) if registry else 0}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.environ.get("TOOL_SYSTEM_HOST", "0.0.0.0"),
        port=int(os.environ.get("TOOL_SYSTEM_PORT", "8004")),
        log_level="info",
    )
