# SPDX-License-Identifier: Apache-2.0
# Standard
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Optional, cast

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.hotness_policy import (
    HOTNESS_PROMOTION_MARGIN,
    HotnessPolicy,
    Tier,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)

DEFAULT_TIER_MANAGER_INTERVAL_SECS = 1.0
DEFAULT_CPU_HIGH_WATERMARK = 0.90
DEFAULT_CPU_LOW_WATERMARK = 0.80
DEFAULT_MAX_ACTIONS_PER_TICK = 8


class TierManager:
    """
    Manage proactive demotion and promotion across storage tiers.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        hotness_policy: HotnessPolicy,
        interval_secs: Optional[float] = None,
        cpu_high_watermark: float = DEFAULT_CPU_HIGH_WATERMARK,
        cpu_low_watermark: float = DEFAULT_CPU_LOW_WATERMARK,
        max_actions_per_tick: int = DEFAULT_MAX_ACTIONS_PER_TICK,
    ) -> None:
        self.storage_manager = storage_manager
        self.hotness_policy = hotness_policy
        self.interval_secs = (
            DEFAULT_TIER_MANAGER_INTERVAL_SECS
            if interval_secs is None
            else interval_secs
        )
        self.cpu_high_watermark = cpu_high_watermark
        self.cpu_low_watermark = cpu_low_watermark
        self.max_actions_per_tick = max_actions_per_tick

        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._state_lock = Lock()

    def start(self) -> None:
        """
        Start the background tier-management loop.
        """
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = Thread(
                target=self._run_loop,
                name="storage-tier-manager",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """
        Stop the background tier-management loop.
        """
        thread: Optional[Thread]
        with self._state_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
            self._thread = None

        if thread.is_alive():
            thread.join(timeout=10.0)

    def is_running(self) -> bool:
        """
        Report whether the tier-manager thread is alive.
        """
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> None:
        """
        Run one management iteration.
        """
        self.hotness_policy.refresh()
        self.evict_cpu_until_below_watermark()
        self.maybe_promote_disk()

    def evict_cpu_until_below_watermark(self) -> None:
        """
        Proactively demote cold CPU keys until CPU usage drops below watermark.
        """
        cpu_backend = self._get_cpu_backend()
        if cpu_backend is None:
            return

        capacity_bytes = cpu_backend.get_capacity_bytes()
        if capacity_bytes <= 0:
            return

        usage_bytes = cpu_backend.get_usage_bytes()
        if usage_bytes <= capacity_bytes * self.cpu_high_watermark:
            return

        target_bytes = int(capacity_bytes * self.cpu_low_watermark)
        candidates = self.hotness_policy.select_coldest(
            Tier.CPU,
            limit=self.max_actions_per_tick,
        )

        for key in candidates:
            if self.demote_key(key):
                usage_bytes = cpu_backend.get_usage_bytes()
                if usage_bytes <= target_bytes:
                    break

    def maybe_promote_disk(self) -> None:
        """
        Promote hot disk-only keys into CPU when they beat the coldest CPU keys.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return

        capacity_bytes = cpu_backend.get_capacity_bytes()
        if capacity_bytes <= 0:
            return

        usage_bytes = cpu_backend.get_usage_bytes()
        if usage_bytes >= capacity_bytes * self.cpu_high_watermark:
            return

        disk_candidates = self.hotness_policy.select_hottest(
            Tier.DISK,
            limit=self.max_actions_per_tick,
            require_absent_in=Tier.CPU,
        )
        if not disk_candidates:
            return

        cpu_coldest = self.hotness_policy.select_coldest(Tier.CPU, limit=1)
        cpu_floor_score = (
            self.hotness_policy.get_score(cpu_coldest[0]) if cpu_coldest else 0.0
        )

        for key in disk_candidates:
            if usage_bytes >= capacity_bytes * self.cpu_high_watermark:
                break
            if not self._should_promote(key, cpu_floor_score):
                continue
            if self.promote_key(key):
                usage_bytes = cpu_backend.get_usage_bytes()

    def promote_key(self, key: CacheEngineKey) -> bool:
        """
        Promote ``key`` from disk into CPU.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return False

        memory_obj = disk_backend.get_blocking(key)
        if memory_obj is None:
            self.hotness_policy.mark_resident(key, Tier.DISK, False)
            return False

        already_in_cpu = cpu_backend.contains(key)
        cpu_backend.submit_put_task(key, memory_obj)
        if not already_in_cpu and cpu_backend.contains(key):
            self.hotness_policy.mark_resident(key, Tier.CPU, True)

        memory_obj.ref_count_down()
        return True

    def demote_key(self, key: CacheEngineKey) -> bool:
        """
        Demote ``key`` from CPU into disk.

        If the key already exists on disk, the demotion completes immediately by
        removing the CPU copy. Otherwise the key is first persisted to disk and
        the CPU copy is removed only after the disk write completes.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return False

        state = self.hotness_policy.get_state(key)
        if state is None or Tier.CPU not in state.resident_tiers:
            return False

        if hasattr(disk_backend, "contains") and disk_backend.contains(key):
            if cpu_backend.remove(key):
                self.hotness_policy.mark_resident(key, Tier.CPU, False)
                self.hotness_policy.mark_resident(key, Tier.DISK, True)
                return True
            return False

        if not hasattr(cpu_backend, "get_blocking") or not hasattr(
            disk_backend, "submit_put_task"
        ):
            return False

        memory_obj = cpu_backend.get_blocking(key)
        if memory_obj is None:
            if not cpu_backend.contains(key):
                self.hotness_policy.mark_resident(key, Tier.CPU, False)
            return False

        def _complete_demote(completed_key: CacheEngineKey) -> None:
            self.hotness_policy.mark_resident(completed_key, Tier.DISK, True)
            if cpu_backend.remove(completed_key) or not cpu_backend.contains(
                completed_key
            ):
                self.hotness_policy.mark_resident(completed_key, Tier.CPU, False)
            memory_obj.ref_count_down()

        put_future = disk_backend.submit_put_task(
            key,
            memory_obj,
            on_complete_callback=_complete_demote,
        )
        if put_future is None:
            memory_obj.ref_count_down()
            return False

        return True

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("TierManager iteration failed")
            self._stop_event.wait(self.interval_secs)

    def _should_promote(self, key: CacheEngineKey, cpu_floor_score: float) -> bool:
        disk_score = self.hotness_policy.get_score(key)
        return disk_score > cpu_floor_score + HOTNESS_PROMOTION_MARGIN

    def _get_cpu_backend(self) -> Optional[object]:
        backend = self.storage_manager.storage_backends.get("LocalCPUBackend")
        if backend is None:
            return None
        required_methods = (
            "get_capacity_bytes",
            "get_usage_bytes",
            "remove",
            "contains",
            "get_blocking",
            "submit_put_task",
        )
        if all(hasattr(backend, method_name) for method_name in required_methods):
            return cast(object, backend)
        return None

    def _get_disk_backend(self) -> Optional[object]:
        backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if backend is None:
            return None
        required_methods = ("get_blocking", "submit_put_task", "contains")
        if all(hasattr(backend, method_name) for method_name in required_methods):
            return cast(object, backend)
        return None
