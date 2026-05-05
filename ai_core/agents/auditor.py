"""Audit Agent — Large OSS 120B+ (primary), Llama 70B + safety models (support).

Handles: logical validation, compliance checks, risk detection,
hallucination filtering, fact verification.
"""
from __future__ import annotations

from typing import Any, Dict

from ..logging_setup import logger
from ..model_selector import TaskType, get_selector
from ..openrouter.client import get_openrouter_client
from .base import AgentContext, BaseAgent

_AUDIT_SYSTEM = """\
You are an elite audit and validation specialist. Your role is to critically evaluate \
outputs for:
1. Logical consistency and internal contradictions
2. Factual accuracy and hallucination detection
3. Compliance with stated constraints
4. Risk identification and safety concerns
5. Completeness and gap analysis

Always output a structured audit report as JSON:
{
  "pass": true/false,
  "score": 0.0-1.0,
  "issues": [{"type": "...", "severity": "low|medium|high|critical", "detail": "..."}],
  "hallucinations_detected": [...],
  "risks": [...],
  "recommendations": [...],
  "revised_output": "..." (only if you can fix critical issues)
}
"""


class AuditAgent(BaseAgent):
    name = "auditor"
    tier = "complex"
    system_prompt = _AUDIT_SYSTEM

    def __init__(self, client=None) -> None:
        super().__init__(client)
        self._selector = get_selector()

    async def audit(
        self,
        ctx: AgentContext,
        original_request: str,
        output_to_audit: str,
        audit_type: str = "general",
        pass_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Audit an output against the original request."""
        spec, provider = self._selector.select(TaskType.AUDIT, require_tools=False)
        logger.info(f"[auditor] model={spec.id} provider={provider} type={audit_type}")

        prompt = (
            f"Audit type: {audit_type}\n\n"
            f"Original request:\n{original_request}\n\n"
            f"Output to audit:\n{output_to_audit}\n\n"
            f"Provide a thorough audit report."
        )
        messages = self._build_messages(ctx, prompt)

        if provider == "nim":
            result = await self.client.chat(messages, model=spec.id, temperature=0.1, max_tokens=3000)
        else:
            or_client = get_openrouter_client()
            result = await or_client.chat(messages, model=spec.id, temperature=0.1, max_tokens=3000)

        content = result.get("content", "")
        structured = self.extract_json(content) or {}

        score = float(structured.get("score", 0.5))
        passed = structured.get("pass", score >= pass_threshold)

        return {
            "pass": bool(passed),
            "score": score,
            "issues": structured.get("issues", []),
            "hallucinations": structured.get("hallucinations_detected", []),
            "risks": structured.get("risks", []),
            "recommendations": structured.get("recommendations", []),
            "revised_output": structured.get("revised_output"),
            "raw_audit": content,
            "model": spec.id,
            "provider": provider,
        }

    async def validate_code(
        self,
        ctx: AgentContext,
        code: str,
        requirements: str,
    ) -> Dict[str, Any]:
        """Validate code against stated requirements."""
        return await self.audit(
            ctx,
            original_request=requirements,
            output_to_audit=f"```python\n{code}\n```",
            audit_type="code_validation",
        )

    async def detect_hallucinations(
        self,
        ctx: AgentContext,
        claim: str,
        context: str,
    ) -> Dict[str, Any]:
        """Check if a claim is supported by the given context."""
        return await self.audit(
            ctx,
            original_request=f"Context:\n{context}",
            output_to_audit=f"Claim to verify:\n{claim}",
            audit_type="hallucination_detection",
        )

    async def risk_assess(
        self,
        ctx: AgentContext,
        strategy: str,
        domain: str = "financial",
    ) -> Dict[str, Any]:
        """Perform risk assessment on a strategy."""
        return await self.audit(
            ctx,
            original_request=f"Domain: {domain}",
            output_to_audit=strategy,
            audit_type="risk_assessment",
        )
