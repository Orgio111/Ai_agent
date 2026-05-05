"""Circuit breaker for NIM API calls."""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreakerOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Async circuit breaker — lock is held only during state transitions, not the guarded call."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 3,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state.value

    async def call(self, coro):
        """Execute coroutine through the circuit breaker.

        Lock is released before awaiting the coroutine so other callers
        are not blocked during the (potentially long) network call.
        """
        async with self._lock:
            self._maybe_transition()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker OPEN — last failure {time.time() - (self._last_failure_time or 0):.0f}s ago"
                )
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    raise CircuitBreakerOpenError("Half-open call limit reached")
                self._half_open_calls += 1
        # Lock released — coroutine runs without blocking other callers.
        try:
            result = await coro
        except Exception:
            async with self._lock:
                self._record_failure()
            raise
        async with self._lock:
            self._record_success()
        return result

    def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._half_open_calls = 0
                logger.info("Circuit breaker CLOSED (recovered)")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            logger.warning("Circuit breaker OPEN again (half-open probe failed)")
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(f"Circuit breaker OPENED after {self._failure_count} failures")

    def _maybe_transition(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and (time.time() - self._last_failure_time) >= self._recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._success_count = 0
            logger.info("Circuit breaker → HALF_OPEN (probing recovery)")
