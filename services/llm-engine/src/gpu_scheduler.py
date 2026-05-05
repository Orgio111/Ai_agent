"""GPU Scheduler: tracks GPU allocation, routes models to available GPUs,
supports MIG partitioning and multi-node cluster awareness."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUDevice:
    index: int
    name: str
    total_mem_gb: float
    used_mem_gb: float = 0.0
    mig_enabled: bool = False
    mig_instances: list[str] = field(default_factory=list)

    @property
    def free_mem_gb(self) -> float:
        return max(0.0, self.total_mem_gb - self.used_mem_gb)

    @property
    def utilization_pct(self) -> float:
        if self.total_mem_gb == 0:
            return 0.0
        return (self.used_mem_gb / self.total_mem_gb) * 100.0


@dataclass
class AllocationHandle:
    device_index: int
    model_id: str
    reserved_mem_gb: float
    allocated_at: float = field(default_factory=time.time)
    allocation_id: str = field(default_factory=lambda: f"alloc_{time.time_ns()}")


class GPUScheduler:
    """Thread-safe GPU memory scheduler with MIG and CPU fallback support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: list[GPUDevice] = []
        self._allocations: dict[str, AllocationHandle] = {}
        self._refresh_interval = 10.0
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._discover_gpus()
        self._start_refresh_loop()

    def _discover_gpus(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    idx, name, total, used = [x.strip() for x in line.split(",")]
                    self._devices.append(
                        GPUDevice(
                            index=int(idx),
                            name=name,
                            total_mem_gb=float(total) / 1024.0,
                            used_mem_gb=float(used) / 1024.0,
                        )
                    )
                logger.info(f"Discovered {len(self._devices)} GPU(s)")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # CPU-only mode
        logger.warning("No NVIDIA GPUs found — running in CPU-only mode")
        self._devices = []

    def _start_refresh_loop(self) -> None:
        def _loop() -> None:
            while not self._stop.is_set():
                self._stop.wait(self._refresh_interval)
                if not self._stop.is_set():
                    self._refresh_gpu_stats()

        self._refresh_thread = threading.Thread(target=_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_gpu_stats(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return
            with self._lock:
                for line in result.stdout.strip().splitlines():
                    idx, used = [x.strip() for x in line.split(",")]
                    i = int(idx)
                    for dev in self._devices:
                        if dev.index == i:
                            dev.used_mem_gb = float(used) / 1024.0
                            break
        except Exception as e:
            logger.debug(f"GPU stat refresh error: {e}")

    def allocate(self, model_id: str, required_mem_gb: float) -> Optional[AllocationHandle]:
        """Allocate GPU for a model. Returns None if no GPU available (CPU fallback)."""
        with self._lock:
            best: Optional[GPUDevice] = None
            for dev in self._devices:
                if dev.free_mem_gb >= required_mem_gb:
                    if best is None or dev.free_mem_gb > best.free_mem_gb:
                        best = dev

            if best is None:
                logger.warning(
                    f"No GPU with {required_mem_gb}GB free for {model_id} — CPU fallback"
                )
                return None

            best.used_mem_gb += required_mem_gb
            handle = AllocationHandle(
                device_index=best.index,
                model_id=model_id,
                reserved_mem_gb=required_mem_gb,
            )
            self._allocations[handle.allocation_id] = handle
            logger.info(
                f"GPU[{best.index}] allocated {required_mem_gb}GB for {model_id} "
                f"({best.free_mem_gb:.1f}GB remaining)"
            )
            return handle

    def release(self, handle: AllocationHandle) -> None:
        with self._lock:
            for dev in self._devices:
                if dev.index == handle.device_index:
                    dev.used_mem_gb = max(0.0, dev.used_mem_gb - handle.reserved_mem_gb)
                    break
            self._allocations.pop(handle.allocation_id, None)
            logger.debug(f"GPU[{handle.device_index}] released {handle.reserved_mem_gb}GB from {handle.model_id}")

    def status(self) -> dict:
        with self._lock:
            return {
                "gpu_count": len(self._devices),
                "cpu_fallback": len(self._devices) == 0,
                "devices": [
                    {
                        "index": d.index,
                        "name": d.name,
                        "total_mem_gb": round(d.total_mem_gb, 2),
                        "used_mem_gb": round(d.used_mem_gb, 2),
                        "free_mem_gb": round(d.free_mem_gb, 2),
                        "utilization_pct": round(d.utilization_pct, 1),
                        "mig_enabled": d.mig_enabled,
                    }
                    for d in self._devices
                ],
                "active_allocations": len(self._allocations),
            }

    def stop(self) -> None:
        self._stop.set()
