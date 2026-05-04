"""Executor agent: runs the plan, coordinating tools and sub-agents."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..logging_setup import logger
from ..tools import get_registry
from .base import AgentContext, BaseAgent
from .coder import CoderAgent


EXECUTOR_SYSTEM = """You are EXECUTOR, the action agent.
Given a single plan step plus prior step outputs, you produce a concise
execution summary or final user-facing response.

RULES
- Be concise. No filler. Reference concrete results from tool outputs.
- If action='respond', produce the user-facing answer directly.
- Otherwise, summarize what was done in 1-3 sentences.
"""


class ExecutorAgent(BaseAgent):
    name = "executor"
    tier = "balanced"
    system_prompt = EXECUTOR_SYSTEM

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.tools = get_registry()
        self.coder = CoderAgent(client=self.client)

    async def execute_plan(self, ctx: AgentContext, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = plan.get("steps", [])
        outputs: List[Dict[str, Any]] = []
        final_response: Optional[str] = None

        for step in steps:
            step_out = await self._execute_step(ctx, step, outputs)
            outputs.append(step_out)
            logger.info(
                f"step {step.get('id')} action={step.get('action')} "
                f"ok={step_out.get('ok')}"
            )
            if step.get("action") == "respond":
                final_response = step_out.get("content", "")

        if final_response is None:
            # Synthesize a final response from outputs if planner forgot.
            final_response = await self._synthesize(ctx, plan, outputs)

        return {"steps": outputs, "final": final_response}

    async def _execute_step(
        self,
        ctx: AgentContext,
        step: Dict[str, Any],
        prior: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        action = (step.get("action") or "respond").lower()
        if action == "tool":
            tool_name = step.get("tool") or ""
            args = step.get("args") or {}
            res = await self.tools.run(tool_name, **args)
            return {"id": step.get("id"), "action": action, "tool": tool_name,
                    "ok": res.ok, "output": res.output, "error": res.error}

        if action == "code":
            spec = step.get("description", "") + "\nArgs:" + str(step.get("args"))
            code = await self.coder.generate(ctx, spec)
            return {"id": step.get("id"), "action": action, "ok": True,
                    "output": code, "content": code.get("content", "")}

        if action == "think":
            prompt = (
                f"Reason briefly about: {step.get('description', '')}\n"
                f"Prior outputs: {prior[-3:] if prior else []}"
            )
            res = await self.call(ctx, prompt)
            return {"id": step.get("id"), "action": action, "ok": True,
                    "content": res["content"]}

        # respond
        prompt = (
            f"Final user-facing response.\nGoal: {ctx.goal}\n"
            f"Step description: {step.get('description', '')}\n"
            f"Prior outputs (compact): {self._compact(prior)}\n"
            "Answer the user clearly and concisely."
        )
        res = await self.call(ctx, prompt)
        return {"id": step.get("id"), "action": action, "ok": True, "content": res["content"]}

    async def _synthesize(
        self,
        ctx: AgentContext,
        plan: Dict[str, Any],
        outputs: List[Dict[str, Any]],
    ) -> str:
        prompt = (
            f"Goal: {ctx.goal}\n"
            f"Plan rationale: {plan.get('rationale', '')}\n"
            f"Step outputs: {self._compact(outputs)}\n"
            "Write the final user response. Concise. No preamble."
        )
        res = await self.call(ctx, prompt)
        return res["content"]

    @staticmethod
    def _compact(outputs: List[Dict[str, Any]]) -> str:
        rows = []
        for o in outputs[-6:]:
            txt = o.get("content") or o.get("output") or o.get("error") or ""
            if isinstance(txt, (dict, list)):
                txt = str(txt)[:400]
            else:
                txt = str(txt)[:400]
            rows.append(f"#{o.get('id')} {o.get('action')}: {txt}")
        return "\n".join(rows)
