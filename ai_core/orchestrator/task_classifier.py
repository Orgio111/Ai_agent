"""Task classifier: maps user prompts to TaskType using keyword heuristics
and a lightweight model call for ambiguous cases."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..model_selector.capability import TaskType

# Compiled patterns for zero-latency keyword classification
_PATTERNS: List[Tuple[TaskType, re.Pattern]] = [
    (TaskType.PERCEPTION,  re.compile(
        r"\b(image|chart|graph|pdf|ocr|visual|screenshot|diagram|parse|extract|"
        r"read|transcri|audio|video)\b", re.I)),
    (TaskType.RESEARCH,    re.compile(
        r"\b(research|analyz|investigat|study|find|discover|survey|review|"
        r"literature|search|what is|how does|why does|explain)\b", re.I)),
    (TaskType.ENGINEERING, re.compile(
        r"\b(code|implement|build|develop|program|function|class|api|"
        r"algorithm|debug|refactor|optimize|script|backtest|deploy)\b", re.I)),
    (TaskType.AUDIT,       re.compile(
        r"\b(audit|verify|validate|check|review|test|compliance|risk|"
        r"hallucination|correct|accurate|fact.check|assess)\b", re.I)),
    (TaskType.REPORTING,   re.compile(
        r"\b(report|summary|summarize|present|dashboard|visualiz|chart|"
        r"communicate|write|draft|document|cfo|executive|board)\b", re.I)),
    (TaskType.PLANNING,    re.compile(
        r"\b(plan|strategy|roadmap|architect|design|orchestrat|coordinat|"
        r"schedule|sequence|priorit|goal|objective)\b", re.I)),
    (TaskType.REASONING,   re.compile(
        r"\b(reason|think|logic|infer|deduce|conclude|hypothes|compare|"
        r"evaluate|pros.and.cons|trade.?off|decision)\b", re.I)),
]

# Score weights for tie-breaking: prefer more specific task types
_TYPE_WEIGHTS: Dict[TaskType, float] = {
    TaskType.ENGINEERING: 1.2,
    TaskType.RESEARCH:    1.1,
    TaskType.AUDIT:       1.1,
    TaskType.PERCEPTION:  1.0,
    TaskType.REPORTING:   1.0,
    TaskType.PLANNING:    0.9,
    TaskType.REASONING:   0.8,
    TaskType.GENERAL:     0.5,
}


class TaskClassifier:
    """Classifies a text prompt into a TaskType.

    Uses compiled regex for instant classification (< 1ms).
    Falls back to GENERAL for ambiguous short prompts.
    """

    def classify(self, prompt: str) -> TaskType:
        """Return the most likely TaskType for the given prompt."""
        scores: Dict[TaskType, float] = {}
        for task_type, pattern in _PATTERNS:
            matches = pattern.findall(prompt)
            if matches:
                score = len(matches) * _TYPE_WEIGHTS.get(task_type, 1.0)
                scores[task_type] = scores.get(task_type, 0.0) + score

        if not scores:
            return TaskType.GENERAL

        return max(scores, key=lambda t: scores[t])

    def classify_with_scores(self, prompt: str) -> Dict[str, float]:
        """Return all task type scores for debugging/logging."""
        scores: Dict[str, float] = {}
        for task_type, pattern in _PATTERNS:
            matches = pattern.findall(prompt)
            if matches:
                scores[task_type.value] = len(matches) * _TYPE_WEIGHTS.get(task_type, 1.0)
        if not scores:
            scores[TaskType.GENERAL.value] = 1.0
        return scores

    def classify_multi(self, prompt: str, top_k: int = 2) -> List[TaskType]:
        """Return top-k task types (for hybrid multi-agent routing)."""
        scores: Dict[TaskType, float] = {}
        for task_type, pattern in _PATTERNS:
            matches = pattern.findall(prompt)
            if matches:
                scores[task_type] = scores.get(task_type, 0.0) + len(matches) * _TYPE_WEIGHTS.get(task_type, 1.0)

        if not scores:
            return [TaskType.GENERAL]

        ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        return ranked[:top_k]


_classifier: TaskClassifier | None = None


def get_classifier() -> TaskClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TaskClassifier()
    return _classifier
