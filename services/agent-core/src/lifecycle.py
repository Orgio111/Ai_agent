"""Agent Lifecycle Manager: run loops, retries, cancellations, health checks, auto-restart."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import websockets

logger = logging.getLogger(__name__)

PRIORITY_ORDER = ["critic", "planner", "researcher", "optimizer", "executor"]


class LifecycleManager:
    def __init__(self, swarm, broker_url: str) -> None:
        self._swarm = swarm
        self._broker_url = broker_url
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._broker_task: Optional[asyncio.Task] = None
        self._restart_counts: dict[str, int] = {}
        self._max_restarts = 5

    async def start(self) -> None:
        self._running = True
        self._health_task = asyncio.create_task(self._health_loop())
        self._broker_task = asyncio.create_task(self._broker_listener())
        logger.info("Lifecycle manager started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        if self._health_task:
            self._health_task.cancel()
        if self._broker_task:
            self._broker_task.cancel()
        if self._ws:
            await self._ws.close()

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "active_tasks": len(self._tasks),
            "restart_counts": self._restart_counts,
            "broker_connected": self._ws is not None and not self._ws.closed,
        }

    async def _broker_listener(self) -> None:
        """Connect to the Rust broker WebSocket and dispatch agent events."""
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(
                    self._broker_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    logger.info(f"Connected to broker: {self._broker_url}")
                    backoff = 1.0

                    # Subscribe to agent events
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "topic": "agent.task",
                        "subscriber_id": "lifecycle-manager",
                    }))

                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            event = json.loads(message)
                            await self._handle_broker_event(event)
                        except Exception as e:
                            logger.error(f"Event handling error: {e}")

            except Exception as e:
                logger.warning(f"Broker connection lost: {e} — retrying in {backoff}s")
                self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _handle_broker_event(self, event: dict) -> None:
        topic = event.get("topic", "")
        payload = event.get("payload", {})

        if topic == "agent.task":
            task_id = event.get("id", "")
            agent_type = payload.get("agent_type", "executor")

            if task_id in self._tasks and not self._tasks[task_id].done():
                logger.debug(f"Task {task_id} already running, skipping duplicate")
                return

            logger.info(f"Lifecycle: dispatching task {task_id} to {agent_type}")
            coro = self._run_agent_task(task_id, agent_type, payload)
            self._tasks[task_id] = asyncio.create_task(coro)

        elif topic == "agent.cancel":
            task_id = payload.get("task_id", "")
            if task_id in self._tasks:
                self._tasks[task_id].cancel()
                del self._tasks[task_id]
                logger.info(f"Lifecycle: cancelled task {task_id}")

    async def _run_agent_task(
        self, task_id: str, agent_type: str, payload: dict
    ) -> None:
        from .models import AgentRequest
        req = AgentRequest(
            prompt=payload.get("prompt", ""),
            session_id=payload.get("session_id", "lifecycle"),
            context=payload.get("context", {}),
        )
        try:
            result = await self._swarm.run(req)
            await self._publish_result(task_id, result.result)
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self._maybe_restart(task_id, agent_type, payload, str(e))
        finally:
            self._tasks.pop(task_id, None)

    async def _maybe_restart(
        self, task_id: str, agent_type: str, payload: dict, error: str
    ) -> None:
        count = self._restart_counts.get(task_id, 0)
        if count >= self._max_restarts:
            logger.error(f"Task {task_id} exhausted restarts ({self._max_restarts})")
            await self._publish_failure(task_id, error)
            return

        self._restart_counts[task_id] = count + 1
        delay = 2 ** count
        logger.warning(f"Restarting task {task_id} (attempt {count + 1}) in {delay}s")
        await asyncio.sleep(delay)
        coro = self._run_agent_task(task_id, agent_type, payload)
        self._tasks[task_id] = asyncio.create_task(coro)

    async def _publish_result(self, task_id: str, result: str) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send(json.dumps({
                "action": "publish",
                "event": {
                    "topic": "agent.result",
                    "payload": {"task_id": task_id, "result": result[:2000]},
                },
            }))

    async def _publish_failure(self, task_id: str, error: str) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send(json.dumps({
                "action": "publish",
                "event": {
                    "topic": "agent.failure",
                    "payload": {"task_id": task_id, "error": error},
                },
            }))

    async def _health_loop(self) -> None:
        while self._running:
            await asyncio.sleep(30)
            # Clean up completed tasks
            done = [tid for tid, t in self._tasks.items() if t.done()]
            for tid in done:
                self._tasks.pop(tid, None)
            logger.debug(f"Lifecycle health: active_tasks={len(self._tasks)}")
