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

        self.index = self._load_index()
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
        faiss.write_index(self.index, str(self.index_path))
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
            self.index = faiss.IndexFlatIP(self.dim)
            self.records = []
            self._persist()
