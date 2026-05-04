"""Centralized configuration loader.

Reads .env, models.yaml, settings.yaml.  Provides a typed Settings
singleton consumed across the AI core.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env", override=False)


class ModelTier(BaseModel):
    model: str
    temperature: float = 0.5
    max_tokens: int = 2048
    top_p: float = 0.9
    description: str = ""


class EmbeddingCfg(BaseModel):
    model: str
    dim: int = 1024


class Heuristics(BaseModel):
    complex_keywords: List[str] = Field(default_factory=list)
    code_keywords: List[str] = Field(default_factory=list)
    fast_keywords: List[str] = Field(default_factory=list)


class ModelsCfg(BaseModel):
    routing: Dict[str, ModelTier]
    embedding: EmbeddingCfg
    heuristics: Heuristics


class Settings(BaseModel):
    # API keys / endpoints
    nim_api_key: str = Field(default_factory=lambda: os.getenv("NVIDIA_NIM_API_KEY", ""))
    nim_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )

    # Service binding
    ai_core_host: str = Field(default_factory=lambda: os.getenv("AI_CORE_HOST", "0.0.0.0"))
    ai_core_port: int = Field(default_factory=lambda: int(os.getenv("AI_CORE_PORT", "8000")))

    # Memory
    memory_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("MEMORY_DIR", "./data/memory")).resolve()
    )
    short_term_max: int = Field(default_factory=lambda: int(os.getenv("SHORT_TERM_MAX", "50")))
    long_term_dim: int = Field(default_factory=lambda: int(os.getenv("LONG_TERM_DIM", "1024")))

    # Loop
    max_iterations: int = Field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "5")))
    critic_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CRITIC_THRESHOLD", "0.75"))
    )

    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Rust perf service URL (optional)
    rust_perf_url: str = Field(
        default_factory=lambda: os.getenv("RUST_PERF_URL", "http://localhost:7070")
    )

    # Loaded YAML configs
    models: ModelsCfg
    raw_settings: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    models_raw = _load_yaml(CONFIG_DIR / "models.yaml")
    settings_raw = _load_yaml(CONFIG_DIR / "settings.yaml")

    models = ModelsCfg(
        routing={k: ModelTier(**v) for k, v in models_raw.get("routing", {}).items()},
        embedding=EmbeddingCfg(**models_raw.get("embedding", {"model": "nvidia/nv-embedqa-e5-v5"})),
        heuristics=Heuristics(**models_raw.get("heuristics", {})),
    )

    s = Settings(models=models, raw_settings=settings_raw)
    s.memory_dir.mkdir(parents=True, exist_ok=True)
    return s


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
