"""Agent Core: FastAPI service managing the multi-agent swarm,
lifecycle, goal engine, and self-improving loop."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .goal_engine import GoalEngine
from .lifecycle import LifecycleManager
from .models import (
    AgentRequest,
    AgentResponse,
    GoalCreateRequest,
    GoalStatusResponse,
    SwarmStatusResponse,
)
from .swarm import AgentSwarm

swarm: AgentSwarm | None = None
lifecycle: LifecycleManager | None = None
goal_engine: GoalEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global swarm, lifecycle, goal_engine

    nim_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    if not nim_key or nim_key.startswith("nvapi-your"):
        raise RuntimeError("NVIDIA_NIM_API_KEY must be set with a valid API key.")

    broker_url = os.environ.get("BROKER_WS_URL", "ws://localhost:8001/ws")
    memory_url = os.environ.get("MEMORY_URL", "http://localhost:8003")
    tool_url = os.environ.get("TOOL_SYSTEM_URL", "http://localhost:8004")

    swarm = AgentSwarm(
        nim_api_key=nim_key,
        nim_base_url=os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        memory_url=memory_url,
        tool_url=tool_url,
    )
    await swarm.start()

    lifecycle = LifecycleManager(swarm=swarm, broker_url=broker_url)
    await lifecycle.start()

    goal_engine = GoalEngine(swarm=swarm, memory_url=memory_url)
    await goal_engine.start()

    yield

    await goal_engine.stop()
    await lifecycle.stop()
    await swarm.stop()


app = FastAPI(title="JARVIS Agent Core", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/run", response_model=AgentResponse)
async def run_agents(req: AgentRequest) -> AgentResponse:
    assert swarm is not None
    try:
        return await swarm.run(req)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/run/stream")
async def run_stream(req: AgentRequest) -> StreamingResponse:
    assert swarm is not None

    async def generate():
        async for chunk in swarm.run_stream(req):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/goals", response_model=GoalStatusResponse)
async def create_goal(req: GoalCreateRequest) -> GoalStatusResponse:
    assert goal_engine is not None
    return await goal_engine.create(req)


@app.get("/goals/{goal_id}", response_model=GoalStatusResponse)
async def get_goal(goal_id: str) -> GoalStatusResponse:
    assert goal_engine is not None
    result = await goal_engine.get(goal_id)
    if not result:
        raise HTTPException(404, f"Goal {goal_id} not found")
    return result


@app.delete("/goals/{goal_id}")
async def cancel_goal(goal_id: str) -> dict:
    assert goal_engine is not None
    await goal_engine.cancel(goal_id)
    return {"goal_id": goal_id, "status": "cancelled"}


@app.get("/goals")
async def list_goals() -> dict:
    assert goal_engine is not None
    return {"goals": await goal_engine.list_all()}


@app.get("/swarm/status", response_model=SwarmStatusResponse)
async def swarm_status() -> SwarmStatusResponse:
    assert swarm is not None and lifecycle is not None
    return SwarmStatusResponse(
        agents=swarm.agent_statuses(),
        lifecycle=lifecycle.status(),
        active_goals=await goal_engine.active_count() if goal_engine else 0,
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "swarm": swarm is not None,
        "lifecycle": lifecycle is not None,
        "goal_engine": goal_engine is not None,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.environ.get("AGENT_CORE_HOST", "0.0.0.0"),
        port=int(os.environ.get("AGENT_CORE_PORT", "8000")),
        log_level="info",
    )
