"""End-to-end smoke test: orchestrator with stubbed NIM client."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from ai_core.config import reload_settings
from ai_core.orchestrator import Orchestrator


class FakeNIM:
    async def chat(self, messages: List[Dict[str, str]], **kw: Any) -> Dict[str, Any]:
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "PLANNER" in sys_prompt:
            content = (
                '{"goal":"g","rationale":"r","steps":'
                '[{"id":1,"action":"respond","description":"reply","tool":null,'
                '"args":{},"success_criteria":"done"}]}'
            )
        elif "CRITIC" in sys_prompt:
            content = '{"score":0.95,"verdict":"accept","issues":[],"suggestions":[]}'
        else:
            content = "Final response."
        return {"content": content, "model": "fake", "tier": "fake", "usage": {}}

    async def embed(self, texts, model=None):
        return [[0.0] * 4 for _ in texts]

    async def close(self):
        pass


def test_orchestrator_runs_one_iteration(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    try:
        monkeypatch.setenv("MEMORY_DIR", str(tmp))
        monkeypatch.setenv("LONG_TERM_DIM", "4")
        reload_settings()

        from ai_core import nim_client as nim_pkg
        from ai_core import memory as mem_pkg
        from ai_core import orchestrator as orch_pkg
        from ai_core import agents as agents_pkg

        fake = FakeNIM()
        # Replace the client singleton everywhere it's already cached.
        nim_pkg.client._singleton = fake  # type: ignore[attr-defined]
        # Reset cached singletons that may have captured the real client.
        mem_pkg.manager._manager = None  # type: ignore[attr-defined]
        orch_pkg.orchestrator._orchestrator = None  # type: ignore[attr-defined]

        orch = Orchestrator()
        # Replace clients on agents (they captured the real one in __init__).
        orch.planner.client = fake  # type: ignore[assignment]
        orch.executor.client = fake  # type: ignore[assignment]
        orch.executor.coder.client = fake  # type: ignore[assignment]
        orch.coder.client = fake  # type: ignore[assignment]
        orch.critic.client = fake  # type: ignore[assignment]

        result = asyncio.run(orch.run("Say hi"))
        assert result.iterations == 1
        assert result.critic["verdict"] == "accept"
        assert "Final response" in result.final_answer
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
