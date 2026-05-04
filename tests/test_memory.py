"""Short-term memory and FAISS long-term memory unit tests."""
from __future__ import annotations

import os
import shutil
import tempfile

from ai_core.memory.long_term import LongTermMemory
from ai_core.memory.short_term import ShortTermMemory


def test_short_term_fifo_bound():
    s = ShortTermMemory(max_messages=3)
    for i in range(5):
        s.add("sess", "user", f"msg{i}")
    msgs = s.messages("sess")
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg2"
    assert msgs[-1]["content"] == "msg4"


def test_short_term_isolation_per_session():
    s = ShortTermMemory(max_messages=10)
    s.add("a", "user", "hello-a")
    s.add("b", "user", "hello-b")
    assert s.messages("a") == [{"role": "user", "content": "hello-a"}]
    assert s.messages("b") == [{"role": "user", "content": "hello-b"}]


def test_long_term_add_and_search():
    tmp = tempfile.mkdtemp()
    try:
        lt = LongTermMemory(dim=4, persist_dir=tmp)
        lt.add("alpha", [1.0, 0.0, 0.0, 0.0])
        lt.add("beta", [0.0, 1.0, 0.0, 0.0])
        lt.add("gamma", [0.0, 0.0, 1.0, 0.0])

        hits = lt.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert len(hits) == 2
        assert hits[0]["text"] == "alpha"
        assert hits[0]["score"] > 0.99

        # Ensure persistence files exist.
        assert os.path.exists(os.path.join(tmp, "vectors.faiss"))
        assert os.path.exists(os.path.join(tmp, "meta.json"))

        # Reload from disk.
        lt2 = LongTermMemory(dim=4, persist_dir=tmp)
        assert len(lt2) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
