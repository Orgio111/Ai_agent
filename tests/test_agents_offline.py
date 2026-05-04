"""Offline agent tests with a stubbed NIM client - no network required."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import patch

from ai_core.agents import (
    AgentContext,
    CriticAgent,
    ExecutorAgent,
    PlannerAgent,
)


class StubClient:
    """Minimal stand-in returning canned responses based on system prompt."""

    def __init__(self, responses: Dict[str, str]) -> None:
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, messages: List[Dict[str, str]], **kw: Any) -> Dict[str, Any]:
        self.calls.append({"messages": messages, "opts": kw})
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        for keyword, text in self.responses.items():
            if keyword in sys_prompt:
                return {"content": text, "model": "stub", "tier": kw.get("tier", "stub"), "usage": {}}
        return {"content": "ok", "model": "stub", "tier": "stub", "usage": {}}


def test_planner_parses_json():
    plan_json = (
        '{"goal":"x","rationale":"do it","steps":'
        '[{"id":1,"action":"respond","description":"answer","tool":null,'
        '"args":{},"success_criteria":"answered"}]}'
    )
    stub = StubClient({"PLANNER": plan_json})
    p = PlannerAgent(client=stub)  # type: ignore[arg-type]
    plan = asyncio.run(p.plan(AgentContext(goal="x")))
    assert plan["steps"][0]["action"] == "respond"
    assert plan["_meta"]["model"] == "stub"


def test_planner_falls_back_on_garbage():
    stub = StubClient({"PLANNER": "this is not JSON at all"})
    p = PlannerAgent(client=stub)  # type: ignore[arg-type]
    plan = asyncio.run(p.plan(AgentContext(goal="x")))
    assert plan["steps"][0]["action"] == "respond"


def test_critic_parses_score_and_normalizes():
    stub = StubClient({"CRITIC": '{"score":0.9,"issues":[],"suggestions":[]}'})
    c = CriticAgent(client=stub)  # type: ignore[arg-type]
    v = asyncio.run(c.review(AgentContext(goal="x"), {"rationale": "r"}, {"final": "ans"}))
    assert v["verdict"] == "accept"
    assert v["score"] == 0.9


def test_critic_marks_low_score_as_improve():
    stub = StubClient({"CRITIC": '{"score":0.3,"suggestions":["be clearer"]}'})
    c = CriticAgent(client=stub)  # type: ignore[arg-type]
    v = asyncio.run(c.review(AgentContext(goal="x"), {}, {"final": ""}))
    assert v["verdict"] == "improve"
    assert v["score"] == 0.3


def test_executor_handles_respond_step():
    stub = StubClient({"EXECUTOR": "Hello, here is the answer."})
    e = ExecutorAgent(client=stub)  # type: ignore[arg-type]
    plan = {
        "rationale": "answer it",
        "steps": [
            {"id": 1, "action": "respond", "description": "answer", "tool": None, "args": {},
             "success_criteria": "ok"}
        ],
    }
    res = asyncio.run(e.execute_plan(AgentContext(goal="hi"), plan))
    assert res["final"].startswith("Hello")
    assert res["steps"][0]["ok"] is True
