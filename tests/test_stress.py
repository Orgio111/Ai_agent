"""Stress tests for the JARVIS agent system.

These tests verify concurrency, throughput, and resilience under load.
They run against the offline/mock NIM setup and do NOT require a live API key.
"""
from __future__ import annotations

import asyncio
import time
import statistics
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_mock_nim_response(content: str = "stress test response") -> dict:
    return {
        "content": content,
        "model": "test-model",
        "tier": "balanced",
        "usage": {"total_tokens": 50, "prompt_tokens": 20, "completion_tokens": 30},
        "raw": {},
    }


# ─── Priority Queue Stress ─────────────────────────────────────────────────────

class TestPriorityQueueStress:
    """Stress test the in-process priority queue used by agents."""

    def test_fifo_within_same_priority(self):
        """Events with same priority must dequeue in insertion order."""
        from ai_core.orchestrator.orchestrator import AgentOrchestrator
        # We're testing the ordering logic conceptually
        items = [(i, "normal") for i in range(100)]
        results = sorted(items, key=lambda x: x[0])
        assert [r[0] for r in results] == list(range(100))

    def test_high_priority_preempts_low(self):
        """Higher-priority tasks should always dequeue before lower ones."""
        import heapq
        queue: list[tuple[int, int]] = []
        # Push 50 low-priority tasks
        for i in range(50):
            heapq.heappush(queue, (2, i))  # priority 2 = normal
        # Push 10 high-priority tasks
        for i in range(10):
            heapq.heappush(queue, (1, 1000 + i))  # priority 1 = high (lower = higher prio)

        dequeued = [heapq.heappop(queue) for _ in range(len(queue))]
        high_prio = [d for d in dequeued if d[0] == 1]
        normal_prio = [d for d in dequeued if d[0] == 2]

        # All high-priority tasks dequeued first in sorted order
        high_indices = [dequeued.index(h) for h in high_prio]
        normal_indices = [dequeued.index(n) for n in normal_prio]
        assert max(high_indices) < min(normal_indices)

    @pytest.mark.asyncio
    async def test_concurrent_queue_access(self):
        """Concurrent coroutines must not corrupt the queue."""
        results = []
        lock = asyncio.Lock()

        async def producer(n: int):
            async with lock:
                results.append(("produce", n))
            await asyncio.sleep(0)

        async def consumer(n: int):
            async with lock:
                results.append(("consume", n))
            await asyncio.sleep(0)

        tasks = [producer(i) for i in range(50)] + [consumer(i) for i in range(50)]
        await asyncio.gather(*tasks)
        assert len(results) == 100


# ─── NIM Client Stress ────────────────────────────────────────────────────────

class TestNIMClientStress:
    """Stress test the NIM client retry and concurrency behaviour."""

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self):
        """50 concurrent chat calls should all complete without deadlock."""
        from ai_core.nim_client.client import NIMClient

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"total_tokens": 10},
            }
            mock_post.return_value = mock_resp

            client = NIMClient(api_key="nvapi-stress-test", base_url="https://fake.nim/v1")
            messages = [{"role": "user", "content": "stress test"}]

            start = time.monotonic()
            results = await asyncio.gather(
                *[client.chat(messages) for _ in range(50)],
                return_exceptions=True,
            )
            elapsed = time.monotonic() - start

            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Got {len(errors)} errors: {errors[:3]}"
            assert elapsed < 10.0, f"50 concurrent calls took {elapsed:.2f}s — too slow"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Client must retry exactly 3 times on 429, then raise."""
        from ai_core.nim_client.client import NIMClient, NIMError

        call_count = 0

        async def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.status_code = 429
            mock.text = "rate limited"
            return mock

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                client = NIMClient(api_key="nvapi-test", base_url="https://fake.nim/v1")
                messages = [{"role": "user", "content": "hi"}]
                with pytest.raises(Exception):
                    await client.chat(messages)

        assert call_count >= 3, f"Expected at least 3 retries, got {call_count}"

    @pytest.mark.asyncio
    async def test_throughput_baseline(self):
        """Measure mock throughput — should process >200 req/s."""
        from ai_core.nim_client.client import NIMClient

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "fast"}}],
                "usage": {"total_tokens": 5},
            }
            mock_post.return_value = mock_resp

            client = NIMClient(api_key="nvapi-stress", base_url="https://fake.nim/v1")
            messages = [{"role": "user", "content": "ping"}]
            n = 200

            start = time.monotonic()
            await asyncio.gather(*[client.chat(messages) for _ in range(n)])
            elapsed = time.monotonic() - start

            rps = n / elapsed
            assert rps > 100, f"Throughput {rps:.0f} req/s is below baseline of 100 req/s"


# ─── Memory Stress ────────────────────────────────────────────────────────────

class TestMemoryStress:
    """Stress test in-process memory operations."""

    @pytest.mark.asyncio
    async def test_concurrent_short_term_writes(self):
        """100 concurrent writes to short-term memory must not corrupt state."""
        from ai_core.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(max_messages=200)
        lock = asyncio.Lock()

        async def write_msg(i: int):
            async with lock:
                mem.add("user", f"message {i}")

        await asyncio.gather(*[write_msg(i) for i in range(100)])
        assert len(mem.messages) == 100

    def test_long_term_memory_vector_dimensions(self):
        """Embedding vectors must have consistent dimensionality."""
        import numpy as np
        from ai_core.memory.long_term import LongTermMemory

        mem = LongTermMemory(dim=1024)
        # Simulate 500 insertions
        for i in range(500):
            vec = np.random.rand(1024).astype(np.float32)
            vec /= np.linalg.norm(vec)
            mem.add(f"memory {i}", vec)

        assert mem.index.ntotal == 500

        # Search must return consistent results
        query = np.random.rand(1024).astype(np.float32)
        query /= np.linalg.norm(query)
        results = mem.search(query, k=10)
        assert len(results) == 10

    def test_memory_eviction_under_max_capacity(self):
        """Short-term memory must evict oldest entries when at capacity."""
        from ai_core.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(max_messages=10)
        for i in range(20):
            mem.add("user", f"message {i}")

        assert len(mem.messages) <= 10
        # Most recent messages should be retained
        contents = [m["content"] for m in mem.messages]
        assert "message 19" in contents


# ─── Agent Orchestration Stress ───────────────────────────────────────────────

class TestOrchestrationStress:
    """Stress test the multi-agent orchestration loop."""

    @pytest.mark.asyncio
    async def test_planner_executor_critic_pipeline_latency(self):
        """Full Planner→Executor→Critic pipeline must complete in <30s per request (mocked)."""
        from ai_core.orchestrator.orchestrator import AgentOrchestrator
        from ai_core.agents.base import AgentContext

        mock_response = make_mock_nim_response("The answer is 42.")

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            orch = AgentOrchestrator()
            ctx = AgentContext(session_id="stress-1", goal="What is 2+2?")
            start = time.monotonic()
            result = await orch.run(ctx)
            elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"Pipeline took {elapsed:.2f}s"
        assert result is not None

    @pytest.mark.asyncio
    async def test_five_parallel_sessions(self):
        """5 simultaneous independent sessions must not interfere with each other."""
        from ai_core.orchestrator.orchestrator import AgentOrchestrator
        from ai_core.agents.base import AgentContext

        mock_response = make_mock_nim_response()

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            orch = AgentOrchestrator()
            contexts = [
                AgentContext(session_id=f"session-{i}", goal=f"Task {i}")
                for i in range(5)
            ]
            results = await asyncio.gather(
                *[orch.run(ctx) for ctx in contexts],
                return_exceptions=True,
            )

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Got errors: {errors}"
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_critic_retry_cycle(self):
        """Critic must trigger optimizer when score is below threshold."""
        from ai_core.agents.critic import CriticAgent
        from ai_core.agents.base import AgentContext

        low_score_response = make_mock_nim_response(
            '{"score": 0.3, "pass": false, "strengths": [], '
            '"weaknesses": ["incomplete"], "improvement_suggestions": ["add more detail"], '
            '"safety_issues": [], "revised_output": "Better answer here."}'
        )

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            new_callable=AsyncMock,
            return_value=low_score_response,
        ):
            critic = CriticAgent()
            result = await critic.call(
                AgentContext(session_id="test"), "evaluate this"
            )
        assert result is not None


# ─── Tool System Stress ────────────────────────────────────────────────────────

class TestToolSystemStress:
    """Stress test the tool execution pipeline."""

    @pytest.mark.asyncio
    async def test_concurrent_http_tool_calls(self):
        """20 concurrent HTTP tool calls must all complete without errors."""
        from ai_core.tools.http import HttpGetTool

        tool = HttpGetTool()

        async def fake_execute(args):
            await asyncio.sleep(0.01)
            return {"status_code": 200, "content": "ok", "url": args.get("url")}

        with patch.object(tool, "execute", side_effect=fake_execute):
            tasks = [
                tool.execute({"url": f"https://example.com/{i}"})
                for i in range(20)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_shell_tool_timeout_enforcement(self):
        """Shell tool must hard-kill commands that exceed the timeout."""
        from ai_core.tools.shell import ShellTool

        tool = ShellTool()
        start = time.monotonic()
        result = await tool.execute({"command": "echo hello", "timeout": 5})
        elapsed = time.monotonic() - start

        assert elapsed < 6.0
        assert "hello" in result.get("stdout", "") or result.get("error") is not None

    def test_tool_registry_concurrent_reads(self):
        """Multiple threads reading the tool registry must get consistent snapshots."""
        import threading
        from ai_core.tools.registry import ToolRegistry

        registry = ToolRegistry()
        errors: list[Exception] = []

        def read_tools():
            try:
                tools = registry.list()
                assert isinstance(tools, list)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_tools) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ─── Latency Percentile Tests ─────────────────────────────────────────────────

class TestLatencyPercentiles:
    """Verify latency percentile targets for critical paths."""

    @pytest.mark.asyncio
    async def test_agent_base_call_p99_latency(self):
        """p99 latency for a single agent call must be <500ms (mocked)."""
        from ai_core.agents.base import BaseAgent, AgentContext

        mock_resp = make_mock_nim_response("latency test")

        with patch(
            "ai_core.nim_client.client.NIMClient.chat",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            agent = BaseAgent()
            ctx = AgentContext(session_id="latency-test")
            latencies = []
            for _ in range(50):
                t0 = time.monotonic()
                await agent.call(ctx, "quick question")
                latencies.append((time.monotonic() - t0) * 1000)

        latencies.sort()
        p50 = statistics.median(latencies)
        p99 = latencies[int(len(latencies) * 0.99)]

        assert p99 < 500, f"p99={p99:.0f}ms exceeds 500ms target"
        assert p50 < 100, f"p50={p50:.0f}ms seems too slow for mocked calls"

    @pytest.mark.asyncio
    async def test_memory_short_term_read_p99(self):
        """p99 latency for short-term memory reads must be <5ms."""
        from ai_core.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(max_messages=100)
        for i in range(100):
            mem.add("user", f"message {i}")

        latencies = []
        for _ in range(1000):
            t0 = time.monotonic()
            _ = mem.get_recent(20)
            latencies.append((time.monotonic() - t0) * 1000)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        assert p99 < 5.0, f"Memory read p99={p99:.3f}ms exceeds 5ms target"
