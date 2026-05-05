"""Self-Improvement Loop — continuous performance evaluation and adaptation.

The system monitors its own outputs, identifies failure patterns,
updates routing preferences, and refines prompts over time.

Improvement axes:
  1. Model performance tracking (latency, quality score per task type)
  2. Prompt refinement (detect recurring weaknesses → update system prompts)
  3. Routing adaptation (downweight consistently poor models)
  4. Pattern library (store successful reasoning chains for reuse)
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_setup import logger
from ..model_selector.capability import TaskType
from ..model_selector.selector import get_selector


@dataclass
class PerformanceRecord:
    model_id: str
    task_type: str
    quality_score: float      # 0.0 – 1.0 (from audit agent)
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    notes: str = ""


@dataclass
class ImprovementInsight:
    insight_type: str          # "routing" | "prompt" | "pattern"
    description: str
    action: str
    created_at: float = field(default_factory=time.time)
    applied: bool = False


class SelfImprovingLoop:
    """Monitors, analyzes, and adapts the system's own behavior over time."""

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        min_samples: int = 5,
        improvement_interval: float = 300.0,  # 5 minutes
    ) -> None:
        self._records: List[PerformanceRecord] = []
        self._insights: List[ImprovementInsight] = []
        self._persist_path = persist_path
        self._min_samples = min_samples
        self._interval = improvement_interval
        self._loop_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

        # Per-model quality EMA (exponential moving average)
        self._model_quality: Dict[str, float] = {}
        self._model_latency: Dict[str, float] = {}

        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record(
        self,
        model_id: str,
        task_type: TaskType,
        quality_score: float,
        latency_ms: float,
        success: bool = True,
        notes: str = "",
    ) -> None:
        """Record a model's performance on a task."""
        rec = PerformanceRecord(
            model_id=model_id,
            task_type=task_type.value,
            quality_score=quality_score,
            latency_ms=latency_ms,
            success=success,
            notes=notes,
        )
        self._records.append(rec)

        # Update EMAs (alpha = 0.2 for stability)
        alpha = 0.2
        self._model_quality[model_id] = (
            alpha * quality_score + (1 - alpha) * self._model_quality.get(model_id, quality_score)
        )
        self._model_latency[model_id] = (
            alpha * latency_ms + (1 - alpha) * self._model_latency.get(model_id, latency_ms)
        )

        # Mark poor performers as less preferred
        if not success or quality_score < 0.4:
            logger.warning(
                f"[self_improve] poor result: model={model_id} score={quality_score:.2f} "
                f"success={success}"
            )

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    async def analyze(self) -> List[ImprovementInsight]:
        """Analyze recent performance and generate improvement insights."""
        if len(self._records) < self._min_samples:
            return []

        new_insights: List[ImprovementInsight] = []

        # 1. Identify consistently underperforming models
        model_scores: Dict[str, List[float]] = defaultdict(list)
        for r in self._records[-100:]:  # last 100 records
            model_scores[r.model_id].append(r.quality_score)

        for model_id, scores in model_scores.items():
            if len(scores) >= 3:
                avg = sum(scores) / len(scores)
                if avg < 0.5:
                    insight = ImprovementInsight(
                        insight_type="routing",
                        description=f"Model '{model_id}' avg quality={avg:.2f} below threshold",
                        action=f"Deprioritize {model_id} in model selector",
                    )
                    new_insights.append(insight)
                    # Apply: mark model as temporarily unavailable
                    get_selector().mark_unavailable(model_id)
                    logger.info(f"[self_improve] deprioritized '{model_id}' (avg_score={avg:.2f})")

        # 2. Identify best-performing task-model pairs
        task_model_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self._records[-200:]:
            task_model_scores[r.task_type][r.model_id].append(r.quality_score)

        for task, models in task_model_scores.items():
            for model_id, scores in models.items():
                if len(scores) >= 5:
                    avg = sum(scores) / len(scores)
                    if avg > 0.85:
                        insight = ImprovementInsight(
                            insight_type="pattern",
                            description=f"Model '{model_id}' excels at {task} (avg={avg:.2f})",
                            action=f"Prefer {model_id} for {task} tasks",
                        )
                        new_insights.append(insight)

        self._insights.extend(new_insights)
        self._persist()
        return new_insights

    def get_model_ranking(self, task_type: TaskType) -> List[Dict[str, Any]]:
        """Return models ranked by quality for a given task type."""
        task_scores: Dict[str, List[float]] = defaultdict(list)
        for r in self._records:
            if r.task_type == task_type.value:
                task_scores[r.model_id].append(r.quality_score)

        ranking = []
        for model_id, scores in task_scores.items():
            avg_quality = sum(scores) / len(scores)
            avg_latency = self._model_latency.get(model_id, 1000.0)
            ranking.append({
                "model_id": model_id,
                "avg_quality": round(avg_quality, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "sample_count": len(scores),
                # Score = quality / log(latency) — prefer fast+good
                "composite_score": round(avg_quality / (1 + avg_latency / 10000), 4),
            })
        return sorted(ranking, key=lambda m: m["composite_score"], reverse=True)

    def stats(self) -> Dict[str, Any]:
        """Return system-wide performance statistics."""
        if not self._records:
            return {"records": 0}

        recent = self._records[-50:]
        return {
            "total_records": len(self._records),
            "recent_avg_quality": round(sum(r.quality_score for r in recent) / len(recent), 3),
            "recent_avg_latency_ms": round(sum(r.latency_ms for r in recent) / len(recent), 1),
            "success_rate": round(sum(1 for r in recent if r.success) / len(recent), 3),
            "models_tracked": len(self._model_quality),
            "insights_generated": len(self._insights),
        }

    # ------------------------------------------------------------------ #
    # Background loop                                                      #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self._loop_task and not self._loop_task.done():
            return
        self._stop.clear()
        self._loop_task = asyncio.create_task(self._background_loop())
        logger.info("[self_improve] background analysis loop started")

    async def stop(self) -> None:
        self._stop.set()
        if self._loop_task:
            await self._loop_task

    async def _background_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self._interval)
            if not self._stop.is_set():
                try:
                    insights = await self.analyze()
                    if insights:
                        logger.info(f"[self_improve] {len(insights)} new insights generated")
                except Exception as e:
                    logger.error(f"[self_improve] analysis error: {e}")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            data = {
                "records": [asdict(r) for r in self._records[-1000:]],  # keep last 1000
                "insights": [asdict(i) for i in self._insights[-200:]],
                "model_quality": self._model_quality,
                "model_latency": self._model_latency,
            }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"[self_improve] persist failed: {e}")

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._records = [PerformanceRecord(**r) for r in data.get("records", [])]
            self._insights = [ImprovementInsight(**i) for i in data.get("insights", [])]
            self._model_quality = data.get("model_quality", {})
            self._model_latency = data.get("model_latency", {})
            logger.info(f"[self_improve] loaded {len(self._records)} records")
        except Exception as e:
            logger.warning(f"[self_improve] load failed: {e}")
