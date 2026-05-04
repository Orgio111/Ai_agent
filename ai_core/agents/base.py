"""Common base for every agent."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_setup import logger
from ..nim_client import NIMClient, get_nim_client


@dataclass
class AgentContext:
    session_id: str = "default"
    goal: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)
    memory_snippets: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    name: str = "agent"
    tier: str = "balanced"
    system_prompt: str = "You are a helpful AI agent."

    def __init__(self, client: Optional[NIMClient] = None) -> None:
        self.client = client or get_nim_client()

    def _build_messages(self, ctx: AgentContext, user_prompt: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if ctx.memory_snippets:
            msgs.append({
                "role": "system",
                "content": f"Relevant long-term memory:\n{ctx.memory_snippets}",
            })
        msgs.extend(ctx.history)
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    async def call(self, ctx: AgentContext, user_prompt: str, **opts: Any) -> Dict[str, Any]:
        messages = self._build_messages(ctx, user_prompt)
        result = await self.client.chat(messages, tier=self.tier, **opts)
        logger.debug(f"[{self.name}] tier={result.get('tier')} model={result.get('model')}")
        return result

    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """Best-effort JSON extraction from a model response."""
        if not text:
            return None
        # Try direct parse first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # ```json ... ``` fenced block.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass

        # First top-level {...} or [...] block.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
        return None
