"""FAISS-backed long-term vector memory.

Persists an `IndexFlatIP` (cosine via L2-normalized vectors) plus a JSON
sidecar mapping vector ids → text + metadata.  Embeddings are produced
by the NIM embedding model.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..logging_setup import logger


def _try_move_to_gpu(cpu_index: "faiss.Index") -> "faiss.Index":
    """Move a FAISS index to GPU[0] if faiss-gpu is available.

    Searches on GPU are orders of magnitude faster for large indices.
    GPU index is kept in-memory only; persistence always uses the CPU copy.
    Returns the original cpu_index unchanged on any failure.
    """
    try:
        res = faiss.StandardGpuResources()  # type: ignore[attr-defined]
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # type: ignore[attr-defined]
        logger.info("FAISS index moved to GPU for accelerated search")
        return gpu_idx
    except (AttributeError, RuntimeError, Exception):
        return cpu_index


@dataclass
class MemoryRecord:
    id: str
    text: str
    ts: float = field(default_factory=time)
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


class LongTermMemory:
    """Persistent FAISS store with simple JSON sidecar metadata."""

    def __init__(self, dim: int, persist_dir: Optional[Path] = None) -> None:
        self.dim = dim
        self._persist_dir = Path(persist_dir) if persist_dir is not None else None
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = self._persist_dir / "vectors.faiss"
            self.meta_path = self._persist_dir / "meta.json"
        else:
            self.index_path = None  # type: ignore[assignment]
            self.meta_path = None  # type: ignore[assignment]
        self._lock = threading.Lock()

        cpu_index = self._load_index()
        self.index = _try_move_to_gpu(cpu_index)
        self._cpu_index = cpu_index  # kept for persistence (GPU index can't be saved directly)
        self.records: List[MemoryRecord] = self._load_meta()

    def _load_index(self) -> faiss.Index:
        if self.index_path is not None and self.index_path.exists():
            try:
                return faiss.read_index(str(self.index_path))
            except Exception as e:  # pragma: no cover
                logger.warning(f"FAISS index unreadable, rebuilding: {e}")
        return faiss.IndexFlatIP(self.dim)

    def _load_meta(self) -> List[MemoryRecord]:
        if self.meta_path is None or not self.meta_path.exists():
            return []
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return [MemoryRecord(**r) for r in data]
        except Exception as e:  # pragma: no cover
            logger.warning(f"Memory metadata unreadable, resetting: {e}")
            return []

    def _persist(self) -> None:
        if self.index_path is None or self.meta_path is None:
            return
        # Always persist the CPU copy — GPU indexes cannot be written to disk.
        save_index = self._cpu_index if self.index is not self._cpu_index else self.index
        faiss.write_index(save_index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps([asdict(r) for r in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return self.index.ntotal

    def add(
        self,
        text: str,
        embedding: Any,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        arr = np.asarray(embedding, dtype="float32").flatten()
        if len(arr) != self.dim:
            raise ValueError(f"Embedding dim {len(arr)} != expected {self.dim}")
        vec = _normalize(arr.reshape(1, -1))
        rec = MemoryRecord(id=str(uuid.uuid4()), text=text, tags=tags or [], meta=meta or {})
        with self._lock:
            if self.index is not self._cpu_index:
                # Keep CPU copy in sync for persistence.
                self._cpu_index.add(vec)
            self.index.add(vec)
            self.records.append(rec)
            self._persist()
        return rec

    def search(
        self,
        embedding: Any,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        arr = np.asarray(embedding, dtype="float32").flatten()
        if len(arr) != self.dim:
            raise ValueError(f"Embedding dim {len(arr)} != expected {self.dim}")
        q = _normalize(arr.reshape(1, -1))
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.records):
                continue
            if score < min_score:
                continue
            rec = self.records[idx]
            out.append({"score": float(score), **asdict(rec)})
        return out

    def reset(self) -> None:
        with self._lock:
            cpu_index = faiss.IndexFlatIP(self.dim)
            self._cpu_index = cpu_index
            self.index = _try_move_to_gpu(cpu_index)
            self.records = []
            self._persist()
