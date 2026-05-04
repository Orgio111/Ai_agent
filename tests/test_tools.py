"""Tool subsystem behavior."""
from __future__ import annotations

import asyncio

from ai_core.tools import get_registry


def test_registry_lists_default_tools():
    reg = get_registry()
    names = {t["name"] for t in reg.list()}
    assert {"shell", "filesystem", "http", "rust_perf"}.issubset(names)


def test_shell_blocks_unlisted_command():
    reg = get_registry()
    res = asyncio.run(reg.run("shell", command="rm -rf /"))
    assert res.ok is False
    assert "allow-list" in res.error or "disabled" in res.error


def test_shell_runs_allowed_command():
    reg = get_registry()
    res = asyncio.run(reg.run("shell", command="echo hello"))
    # 'echo' is in the default allow list.
    assert res.ok is True
    assert "hello" in res.output["stdout"]


def test_filesystem_sandboxed():
    reg = get_registry()
    # Writing inside the sandbox succeeds.
    res = asyncio.run(reg.run("filesystem", op="write", path="t.txt", content="hi"))
    assert res.ok is True
    res = asyncio.run(reg.run("filesystem", op="read", path="t.txt"))
    assert res.ok and res.output == "hi"

    # Escaping the sandbox fails.
    res = asyncio.run(reg.run("filesystem", op="read", path="../../etc/passwd"))
    assert res.ok is False
