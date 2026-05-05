"""Centralized loguru setup."""
from __future__ import annotations

import sys

from loguru import logger

from .config import get_settings

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    s = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=s.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        backtrace=False,
        diagnose=False,
    )
    _configured = True


__all__ = ["logger", "setup_logging"]
