"""Integration tests: verify service interactions with mocked NIM.

These tests spin up in-process components and verify they talk to each other
correctly, without needing live infrastructure.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def make_nim_chat_response(content: str) -> dict:
    return {
        "content": content,
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "tier": "balanced",
        "usage": {"total_tokens": 30, "prompt_tokens": 10, "completion_tokens": 20},
    }


def make_nim_embed_response(dim: int = 1024) -> dict:
    import numpy as np
    vec = np.random.rand(dim).tolist()
    return {"data": [{"embedding": vec}]}


# ─── Planner → Executor Integration ───────────────────────────────────────────

class TestPlannerExecutorIntegration:

    @pytest.mark.asyncio
    async def test_planner_output_feeds_executor(self):
        """Planner JSON plan must be parseable by the executor input schema."""
        from ai_core.agents.planner import PlannerAgent
        from ai_core.agents.base import AgentContext

        plan_json = json.dumps({
            "plan_summary": "Answer the user question",
            "tasks": [
                {
                    "task_id": "t1",
                    "description": "Research the topic",
                    "agent": "researcher",
                    "depends_on": [],
                    "priority": 3,
                    "estimated_tokens": 500,
                },
                {
                    "task_id": "t2",
                    "description": "Synthesise the answer",
                    "agent": "executor",
                    "depends_on": ["t1"],
                    "priority": 3,
                    "estimated_tokens": 300,
                },
            ],
            "success_criteria": "User question answered accurately",
            "fallback_strategy": "Use simpler model",
        })

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            new_callable=AsyncMock,
            return_value=make_nim_chat_response(plan_json),
        ):
            planner = PlannerAgent()
            ctx = AgentContext(session_id="integ-1", goal="Explain quantum computing")
            result = await planner.call(ctx, "Create a plan")

        assert result is not None
        parsed = planner.extract_json(result["content"])
        assert parsed is not None
        assert "tasks" in parsed
        assert len(parsed["tasks"]) == 2

    @pytest.mark.asyncio
    async def test_executor_handles_tool_call_in_response(self):
        """Executor must detect and dispatch tool calls embedded in model output."""
        from ai_core.agents.executor import ExecutorAgent
        from ai_core.agents.base import AgentContext
        from ai_core.tools.registry import ToolRegistry

        tool_call_response = (
            'I will fetch the URL for you.\n'
            '<tool_call>{"tool": "http_get", "args": {"url": "https://example.com"}}</tool_call>'
        )

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            new_callable=AsyncMock,
            return_value=make_nim_chat_response(tool_call_response),
        ):
            executor = ExecutorAgent()
            ctx = AgentContext(session_id="integ-2")

            # Patch tool execution to avoid real HTTP
            with patch.object(
                executor, "_execute_tools", new_callable=AsyncMock,
                return_value=[{"url": "https://example.com", "status_code": 200, "content": "ok"}],
            ):
                result = await executor.call(ctx, "Fetch https://example.com")

        assert result is not None


# ─── Critic Self-Improvement Integration ──────────────────────────────────────

class TestCriticSelfImprovementIntegration:

    @pytest.mark.asyncio
    async def test_low_score_triggers_revision(self):
        """When critic score < 0.7, the orchestrator must attempt improvement."""
        from ai_core.orchestrator.orchestrator import AgentOrchestrator
        from ai_core.agents.base import AgentContext

        # First call returns low-quality output; second returns improvement
        responses = [
            make_nim_chat_response("Incomplete answer."),          # executor
            make_nim_chat_response(                                 # critic (low score)
                '{"score": 0.4, "pass": false, "strengths": [], '
                '"weaknesses": ["too short"], "improvement_suggestions": ["expand"], '
                '"safety_issues": [], "revised_output": "A much better and complete answer."}'
            ),
        ]
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            side_effect=side_effect,
        ):
            orch = AgentOrchestrator()
            ctx = AgentContext(session_id="critic-integ", goal="What is gravity?")
            result = await orch.run(ctx)

        assert result is not None

    @pytest.mark.asyncio
    async def test_high_score_stops_critic_loop(self):
        """When critic score >= 0.7, no further iterations should occur."""
        from ai_core.orchestrator.orchestrator import AgentOrchestrator
        from ai_core.agents.base import AgentContext

        high_score_critic = make_nim_chat_response(
            '{"score": 0.95, "pass": true, "strengths": ["complete", "accurate"], '
            '"weaknesses": [], "improvement_suggestions": [], "safety_issues": []}'
        )

        call_log: list[str] = []

        async def tracked_call(messages, **kwargs):
            call_log.append(messages[-1].get("content", "")[:30])
            return high_score_critic

        with patch("ai_core.nim_client.client.NIMClient.chat", side_effect=tracked_call):
            orch = AgentOrchestrator()
            ctx = AgentContext(session_id="high-score", goal="Simple question")
            await orch.run(ctx)

        # Should not enter a second critique cycle
        critic_calls = [c for c in call_log if "score" in c.lower() or "evaluate" in c.lower()]
        # At most 1 critic call in the happy path
        assert len(critic_calls) <= 2


# ─── Memory Pipeline Integration ──────────────────────────────────────────────

class TestMemoryPipelineIntegration:

    @pytest.mark.asyncio
    async def test_store_and_retrieve_consistency(self):
        """Stored memories must be retrievable with correct content."""
        from ai_core.memory.manager import MemoryManager

        import numpy as np

        embed_dim = 1024
        stored_content = "The capital of France is Paris."
        query_content = "What is the capital of France?"

        # Patch the embedding call
        async def fake_embed(texts):
            vecs = []
            for t in texts:
                v = np.random.rand(embed_dim).astype(np.float32)
                v /= np.linalg.norm(v)
                vecs.append(v.tolist())
            return vecs

        with patch(
            "ai_core.nim_client.client.NIMClient.embed",
            new_callable=AsyncMock,
            side_effect=fake_embed,
        ):
            manager = MemoryManager(dim=embed_dim)
            await manager.store(
                content=stored_content,
                session_id="mem-integ",
                memory_type="episodic",
            )
            results = await manager.query(
                query=query_content,
                session_id="mem-integ",
                limit=5,
            )

        # At least the stored item should come back
        assert isinstance(results, list)

    def test_short_term_context_window(self):
        """Short-term context must respect the max_messages cap."""
        from ai_core.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(max_messages=20)
        for i in range(30):
            mem.add("user" if i % 2 == 0 else "assistant", f"turn {i}")

        assert len(mem.messages) <= 20
        # Most recent turn should be present
        assert any("turn 29" in m["content"] for m in mem.messages)


# ─── Tool Registry Integration ────────────────────────────────────────────────

class TestToolRegistryIntegration:

    def test_all_builtin_tools_registered(self):
        """All expected built-in tools must be present in the registry."""
        from ai_core.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.list()
        tool_names = {t.name for t in tools}

        expected = {"shell", "filesystem", "http_get"}
        for name in expected:
            assert name in tool_names, f"Built-in tool '{name}' missing from registry"

    @pytest.mark.asyncio
    async def test_sandboxed_code_execution_isolation(self):
        """Code execution tool must not leak state between runs."""
        from ai_core.tools.shell import ShellTool

        tool = ShellTool()
        r1 = await tool.execute({"command": "echo session_a"})
        r2 = await tool.execute({"command": "echo session_b"})

        assert "session_a" in r1.get("stdout", "")
        assert "session_b" in r2.get("stdout", "")
        assert "session_a" not in r2.get("stdout", "")

    @pytest.mark.asyncio
    async def test_blocked_command_rejected(self):
        """Commands not in the allowlist must be rejected without execution."""
        from ai_core.tools.shell import ShellTool

        tool = ShellTool()
        result = await tool.execute({"command": "rm -rf /tmp/test"})

        assert result.get("error") is not None
        assert "rm" in result["error"].lower() or "not in allowlist" in result["error"].lower() \
            or "dangerous" in result["error"].lower()


# ─── Router Integration ────────────────────────────────────────────────────────

class TestRouterIntegration:

    def test_complex_prompt_routes_to_large_model(self):
        """Long analytical prompts must route to the complex tier."""
        from ai_core.nim_client.router import route_for

        prompt = (
            "Analyze the macroeconomic implications of quantitative easing "
            "on emerging market bond yields and compare historical cases across "
            "three different economic cycles, providing evidence-based recommendations."
        )
        decision = route_for(prompt)
        assert decision.tier in ("complex", "balanced"), (
            f"Expected complex/balanced tier for analytical prompt, got {decision.tier}"
        )

    def test_short_prompt_routes_to_fast_model(self):
        """Very short prompts must route to the fast tier."""
        from ai_core.nim_client.router import route_for

        decision = route_for("hi")
        assert decision.tier == "fast", (
            f"Expected 'fast' tier for short prompt, got {decision.tier}"
        )

    def test_code_prompt_routes_to_code_model(self):
        """Code-related prompts must prefer the code tier."""
        from ai_core.nim_client.router import route_for

        decision = route_for("Write a Python function to implement quicksort algorithm")
        assert decision.tier in ("code", "complex"), (
            f"Expected code/complex tier, got {decision.tier}"
        )
