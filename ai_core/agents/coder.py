"""Coder agent: writes high-quality production code on demand."""
from __future__ import annotations

import re
from typing import Any, Dict

from .base import AgentContext, BaseAgent


CODER_SYSTEM = """You are CODER, a senior software engineer agent.
Write production-grade code: complete, runnable, well-typed, no TODOs,
no placeholders. Prefer the language explicitly requested; otherwise pick
the most appropriate. When asked, output a single fenced code block plus
a one-paragraph explanation underneath.
"""


class CoderAgent(BaseAgent):
    name = "coder"
    tier = "code"
    system_prompt = CODER_SYSTEM

    async def generate(self, ctx: AgentContext, spec: str, language: str = "") -> Dict[str, Any]:
        prompt = f"SPEC:\n{spec}\n"
        if language:
            prompt += f"LANGUAGE: {language}\n"
        prompt += "Produce the complete implementation now."
        res = await self.call(ctx, prompt)
        content = res["content"]
        code = self._extract_code(content)
        return {
            "content": content,
            "code": code,
            "language": language or self._guess_language(content),
            "model": res.get("model"),
            "tier": res.get("tier"),
        }

    @staticmethod
    def _extract_code(text: str) -> str:
        m = re.search(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _guess_language(text: str) -> str:
        m = re.search(r"```([a-zA-Z0-9_+\-]+)", text)
        return m.group(1).lower() if m else ""
