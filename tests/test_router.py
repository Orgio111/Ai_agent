"""Routing heuristics should pick the right tier for representative prompts."""
from __future__ import annotations

from ai_core.nim_client.router import ModelRouter


def test_explicit_tier_hint_wins():
    r = ModelRouter()
    d = r.route("anything", tier_hint="fast")
    assert d.tier == "fast"
    assert "explicit" in d.reason


def test_complex_long_prompt_routes_to_complex():
    r = ModelRouter()
    long = "design a distributed system " * 80
    d = r.route(long)
    assert d.tier == "complex"


def test_code_keyword_routes_to_code():
    r = ModelRouter()
    d = r.route("Please implement a function that sorts a list.")
    assert d.tier == "code"


def test_short_greeting_routes_fast():
    r = ModelRouter()
    d = r.route("hi")
    assert d.tier == "fast"


def test_default_balanced():
    r = ModelRouter()
    d = r.route("Tell me about apples.")
    assert d.tier == "balanced"
