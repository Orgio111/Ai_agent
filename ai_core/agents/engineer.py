"""Engineering Agent — GLM/Qwen Engineering (primary), Qwen Coder swarm (support).

Handles: code generation, backtesting, API building, optimization loops,
autonomous execution with tool calling.
"""
from __future__ import annotations

from typing import Any, Dict

from ..logging_setup import logger
from ..model_selector import TaskType, get_selector
from ..openrouter.client import get_openrouter_client
from ..tools.registry import get_registry
from .base import AgentContext, BaseAgent

_ENGINEER_SYSTEM = """\
You are an elite software engineer and autonomous execution agent. You:
1. Write clean, tested, production-ready code
2. Use tool calls to execute code and validate results
3. Iterate until the solution is correct and optimized
4. Think through edge cases and error handling
5. Follow ReAct: Reason → Act → Observe → Repeat

When generating code, always:
- Include error handling
- Add type hints
- Verify logic before execution
- Output structured result: {"code": "...", "explanation": "...", "tests": [...]}
"""


class EngineerAgent(BaseAgent):
    name = "engineer"
    tier = "code"
    system_prompt = _ENGINEER_SYSTEM

    def __init__(self, client=None) -> None:
        super().__init__(client)
        self._selector = get_selector()
        self._registry = get_registry()

    async def generate_code(
        self,
        ctx: AgentContext,
        task: str,
        language: str = "python",
        execute: bool = False,
    ) -> Dict[str, Any]:
        """Generate code for a task, optionally execute and iterate."""
        spec, provider = self._selector.select(TaskType.ENGINEERING, require_tools=execute)
        logger.info(f"[engineer] model={spec.id} provider={provider} execute={execute}")

        prompt = (
            f"Write {language} code for the following task:\n\n{task}\n\n"
            f"Output as JSON: {{\"code\": \"...\", \"explanation\": \"...\", \"tests\": [...]}}"
        )
        messages = self._build_messages(ctx, prompt)

        if provider == "nim":
            result = await self.client.chat(messages, model=spec.id, temperature=0.1, max_tokens=4096)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=spec.id, temperature=0.1, max_tokens=4096)

        content = result.get("content", "")
        structured = self.extract_json(content) or {"code": content}
        code = structured.get("code", content)

        execution_result = None
        if execute and code:
            execution_result = await self._execute_code(code, language)
            # If execution fails, attempt self-correction
            if execution_result.get("error"):
                structured, execution_result = await self._self_correct(
                    ctx, task, code, execution_result["error"], spec.id, provider
                )

        return {
            "task": task,
            "language": language,
            "code": code,
            "explanation": structured.get("explanation", ""),
            "tests": structured.get("tests", []),
            "execution": execution_result,
            "model": spec.id,
            "provider": provider,
        }

    async def _execute_code(self, code: str, language: str) -> Dict[str, Any]:
        """Execute code via the shell tool (sandboxed)."""
        if language == "python":
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            try:
                result = await self._registry.run("shell", command=f"python3 {tmp_path}", timeout=30.0)
                return {
                    "stdout": result.output.get("stdout", "") if result.output else "",
                    "stderr": result.output.get("stderr", "") if result.output else "",
                    "returncode": result.output.get("returncode", -1) if result.output else -1,
                    "error": result.error if not result.ok else None,
                }
            finally:
                os.unlink(tmp_path)
        return {"error": f"Execution not supported for language: {language}"}

    async def _self_correct(
        self,
        ctx: AgentContext,
        original_task: str,
        failed_code: str,
        error: str,
        model_id: str,
        provider: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """ReAct self-correction loop on execution failure."""
        correction_prompt = (
            f"This code failed with error:\n\n{error}\n\n"
            f"Original code:\n```python\n{failed_code}\n```\n\n"
            f"Fix the bug and output corrected JSON: {{\"code\": \"...\", \"explanation\": \"...\"}}"
        )
        messages = self._build_messages(ctx, correction_prompt)
        if provider == "nim":
            result = await self.client.chat(messages, model=model_id, temperature=0.1, max_tokens=4096)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=model_id, temperature=0.1, max_tokens=4096)

        corrected = self.extract_json(result.get("content", "")) or {"code": result.get("content", "")}
        exec_result = await self._execute_code(corrected.get("code", ""), "python")
        return corrected, exec_result

    async def backtest(
        self,
        ctx: AgentContext,
        strategy_description: str,
        data_description: str,
    ) -> Dict[str, Any]:
        """Generate and execute a backtesting strategy."""
        task = (
            f"Write a Python backtesting implementation for:\n"
            f"Strategy: {strategy_description}\n"
            f"Data: {data_description}\n"
            f"Include: entry/exit logic, performance metrics (Sharpe, max drawdown, CAGR)."
        )
        return await self.generate_code(ctx, task, language="python", execute=True)
