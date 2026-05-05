"""GPU + CPU scheduler: tracks allocation across both GPU devices and CPU worker
pool so inference tasks can run on whichever resource is most appropriate —
GPU for large/complex models, CPU for lightweight/embedding tasks — simultaneously.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

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
class CPUWorkerPool:
    """Tracks concurrent CPU inference slots."""
    total_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    active_workers: int = 0

    @property
    def free_workers(self) -> int:
        return max(0, self.total_workers - self.active_workers)

    @property
    def utilization_pct(self) -> float:
        if self.total_workers == 0:
            return 0.0
        return (self.active_workers / self.total_workers) * 100.0


@dataclass
class AllocationHandle:
    device_index: int          # GPU index, or -1 for CPU
    device_type: Literal["gpu", "cpu"]
    model_id: str
    reserved_mem_gb: float
    allocated_at: float = field(default_factory=time.time)
    allocation_id: str = field(default_factory=lambda: f"alloc_{time.time_ns()}")

    @property
    def on_gpu(self) -> bool:
        return self.device_type == "gpu"


class GPUScheduler:
    """Thread-safe scheduler for GPU devices AND CPU worker pool.

    GPU and CPU allocations are tracked independently so both can be
    utilised simultaneously — e.g. GPU for LLM inference while CPU
    handles embedding or fast-tier requests in parallel.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: list[GPUDevice] = []
        self._cpu_pool = CPUWorkerPool()
        self._allocations: dict[str, AllocationHandle] = {}
        self._refresh_interval = 10.0
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._discover_gpus()
        self._start_refresh_loop()

    # ------------------------------------------------------------------ #
    # Discovery / refresh                                                  #
    # ------------------------------------------------------------------ #

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
                logger.info(
                    f"Discovered {len(self._devices)} GPU(s) + "
                    f"{self._cpu_pool.total_workers} CPU worker(s)"
                )
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        logger.warning(
            f"No NVIDIA GPUs found — hybrid mode: CPU-only for inference "
            f"({self._cpu_pool.total_workers} workers)"
        )

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

    # ------------------------------------------------------------------ #
    # Allocation                                                           #
    # ------------------------------------------------------------------ #

    def allocate(
        self,
        model_id: str,
        required_mem_gb: float,
        prefer_cpu: bool = False,
    ) -> Optional[AllocationHandle]:
        """Allocate a device for the given model.

        Args:
            model_id:        Model identifier (for logging).
            required_mem_gb: GPU VRAM needed; ignored when allocating CPU.
            prefer_cpu:      If True, skip GPU and use the CPU pool directly
                             (suitable for small/embedding workloads so GPU
                             stays free for heavy inference).

        Returns:
            AllocationHandle with device_type="gpu" or "cpu", or None if
            both pools are exhausted.
        """
        with self._lock:
            if not prefer_cpu and self._devices:
                handle = self._allocate_gpu(model_id, required_mem_gb)
                if handle is not None:
                    return handle
                logger.info(
                    f"No GPU with {required_mem_gb:.1f}GB free for '{model_id}' "
                    "— routing to CPU pool"
                )

            return self._allocate_cpu(model_id)

    def _allocate_gpu(self, model_id: str, required_mem_gb: float) -> Optional[AllocationHandle]:
        best: Optional[GPUDevice] = None
        for dev in self._devices:
            if dev.free_mem_gb >= required_mem_gb:
                if best is None or dev.free_mem_gb > best.free_mem_gb:
                    best = dev
        if best is None:
            return None

        best.used_mem_gb += required_mem_gb
        handle = AllocationHandle(
            device_index=best.index,
            device_type="gpu",
            model_id=model_id,
            reserved_mem_gb=required_mem_gb,
        )
        self._allocations[handle.allocation_id] = handle
        logger.info(
            f"GPU[{best.index}] allocated {required_mem_gb:.1f}GB for '{model_id}' "
            f"({best.free_mem_gb:.1f}GB remaining)"
        )
        return handle

    def _allocate_cpu(self, model_id: str) -> Optional[AllocationHandle]:
        self._cpu_pool.active_workers += 1
        handle = AllocationHandle(
            device_index=-1,
            device_type="cpu",
            model_id=model_id,
            reserved_mem_gb=0.0,
        )
        self._allocations[handle.allocation_id] = handle
        logger.debug(
            f"CPU worker allocated for '{model_id}' "
            f"({self._cpu_pool.free_workers} free)"
        )
        return handle

    def release(self, handle: AllocationHandle) -> None:
        with self._lock:
            if handle.device_type == "gpu":
                for dev in self._devices:
                    if dev.index == handle.device_index:
                        dev.used_mem_gb = max(0.0, dev.used_mem_gb - handle.reserved_mem_gb)
                        break
                logger.debug(
                    f"GPU[{handle.device_index}] released {handle.reserved_mem_gb:.1f}GB "
                    f"from '{handle.model_id}'"
                )
            else:
                self._cpu_pool.active_workers = max(0, self._cpu_pool.active_workers - 1)
                logger.debug(f"CPU worker released from '{handle.model_id}'")
            self._allocations.pop(handle.allocation_id, None)

    # ------------------------------------------------------------------ #
    # Observability                                                        #
    # ------------------------------------------------------------------ #

    def status(self) -> dict:
        with self._lock:
            return {
                "gpu_count": len(self._devices),
                "cpu_only": len(self._devices) == 0,
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
                "cpu_pool": {
                    "total_workers": self._cpu_pool.total_workers,
                    "active_workers": self._cpu_pool.active_workers,
                    "free_workers": self._cpu_pool.free_workers,
                    "utilization_pct": round(self._cpu_pool.utilization_pct, 1),
                },
                "active_allocations": len(self._allocations),
            }

    def stop(self) -> None:
        self._stop.set()
