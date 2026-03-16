# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from threading import Event, Lock, Thread
from time import time
from typing import TYPE_CHECKING, Optional, Sequence, cast

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.hotness_policy import HotnessPolicy


class Tier(Enum):
    CPU = auto()
    DISK = auto()
    REMOTE = auto()

PROMOTION_MARGIN = 0.05

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
    Manage periodic disk-to-CPU promotion across storage tiers.

    CPU demotion is event-driven through ``ensure_cpu_headroom()`` via the
    LocalCPU pressure handler. The background loop only refreshes hotness aging
    and evaluates replacement promotions.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        hotness_policy: Optional[HotnessPolicy] = None,
        interval_secs: Optional[float] = None,
        cpu_high_watermark: float = DEFAULT_CPU_HIGH_WATERMARK,
        cpu_low_watermark: float = DEFAULT_CPU_LOW_WATERMARK,
        max_actions_per_tick: int = DEFAULT_MAX_ACTIONS_PER_TICK,
    ) -> None:
        self.storage_manager = storage_manager
        self.hotness_policy = hotness_policy or HotnessPolicy()
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
        self._pressure_lock = Lock()

        # Cached backend references (populated lazily).
        self._cached_cpu_backend: Optional[object] = None
        self._cached_disk_backend: Optional[object] = None

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

    # ------------------------------------------------------------------
    # Hotness observation proxies — StorageManager delegates here
    # ------------------------------------------------------------------

    def observe_store(self, keys: Sequence[CacheEngineKey]) -> None:
        now = time()
        for prefix_pos, key in enumerate(keys):
            self.hotness_policy.observe_store(key, prefix_pos, now=now)

    def on_hit(self, key: CacheEngineKey) -> None:
        self.hotness_policy.on_hit(key)

    def mark_resident(
        self, key: CacheEngineKey, tier: Tier, present: bool
    ) -> None:
        self.hotness_policy.mark_resident(key, tier, present)

    def clear_tier(self, tier: Tier) -> None:
        self.hotness_policy.clear_tier(tier)

    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """
        Run one promotion-management iteration.
        """
        with self._pressure_lock:
            self.hotness_policy.refresh()
            self._maybe_replace_promote_disk_locked()

    def maybe_replace_promote_disk(self) -> None:
        """
        Promote hot disk-only keys by replacing colder CPU keys.
        """
        with self._pressure_lock:
            self._maybe_replace_promote_disk_locked()

    def _maybe_replace_promote_disk_locked(self) -> None:
        cpu_backend = self._get_cpu_backend()
        if cpu_backend is None:
            return

        disk_candidates = self.hotness_policy.select_hottest(
            Tier.DISK,
            limit=self.max_actions_per_tick,
            require_absent_in=Tier.CPU,
        )
        if not disk_candidates:
            return

        cpu_candidates = self.hotness_policy.select_coldest(
            Tier.CPU,
            limit=self.max_actions_per_tick,
        )
        if not cpu_candidates:
            return

        used_cpu_keys: set[CacheEngineKey] = set()

        for disk_key in disk_candidates:
            disk_score = self.hotness_policy.get_score(disk_key)
            victim_key: Optional[CacheEngineKey] = None

            for cpu_key in cpu_candidates:
                if cpu_key in used_cpu_keys:
                    continue
                state = self.hotness_policy.get_state(cpu_key)
                if state is None or Tier.DISK not in state.resident_tiers:
                    continue
                cpu_score = self.hotness_policy.get_score(cpu_key)
                if disk_score <= cpu_score + PROMOTION_MARGIN:
                    continue
                victim_key = cpu_key
                break

            if victim_key is None:
                continue

            if self.replace_promote_key(disk_key, victim_key):
                used_cpu_keys.add(victim_key)

    def ensure_cpu_headroom(self) -> bool:
        """
        Synchronously demote CPU-resident keys until usage drops below the low
        watermark.

        This method is the event-driven CPU demotion path used by the local CPU
        pressure handler.

        Returns:
            True if any CPU headroom was created, otherwise False.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return False

        capacity_bytes = cpu_backend.get_capacity_bytes()
        if capacity_bytes <= 0:
            return False

        target_bytes = int(capacity_bytes * self.cpu_low_watermark)
        relieved = False

        with self._pressure_lock:
            usage_bytes = cpu_backend.get_usage_bytes()
            if usage_bytes <= target_bytes:
                return False

            while usage_bytes > target_bytes:
                candidates = self.hotness_policy.select_coldest(
                    Tier.CPU,
                    limit=self.max_actions_per_tick,
                )
                if not candidates:
                    break

                made_progress = False
                for key in candidates:
                    if self.demote_key(key, blocking=True):
                        relieved = True
                        made_progress = True
                        usage_bytes = cpu_backend.get_usage_bytes()
                        if usage_bytes <= target_bytes:
                            break

                if not made_progress:
                    break

        return relieved

    def promote_key(self, key: CacheEngineKey) -> bool:
        """
        Promote ``key`` from disk into CPU.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return False

        memory_obj = self._load_disk_memory_obj(disk_backend, key)
        if memory_obj is None:
            return False

        return self._store_loaded_key_in_cpu(cpu_backend, key, memory_obj)

    def replace_promote_key(
        self,
        disk_key: CacheEngineKey,
        victim_cpu_key: CacheEngineKey,
    ) -> bool:
        """
        Replace a cold CPU-resident key with a hotter disk-only key.

        The current implementation only replaces victims that already have a
        disk copy so the victim can be removed from CPU immediately.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return False

        memory_obj = self._load_disk_memory_obj(disk_backend, disk_key)
        if memory_obj is None:
            return False

        victim_state = self.hotness_policy.get_state(victim_cpu_key)
        if victim_state is None or Tier.DISK not in victim_state.resident_tiers:
            self._release_memory_obj(memory_obj)
            return False
        if not self._remove_cpu_if_evictable(cpu_backend, victim_cpu_key):
            self._release_memory_obj(memory_obj)
            return False

        self.hotness_policy.mark_resident(victim_cpu_key, Tier.CPU, False)
        return self._store_loaded_key_in_cpu(cpu_backend, disk_key, memory_obj)

    def demote_key(self, key: CacheEngineKey, blocking: bool = False) -> bool:
        """
        Demote ``key`` from CPU into disk.

        If the key already exists on disk, the demotion completes immediately by
        removing the CPU copy. Otherwise the key is first persisted to disk and
        the CPU copy is removed only after the disk write completes.

        Args:
            key: Cache key to demote.
            blocking: Whether to wait for the disk put to finish before
                returning.

        Returns:
            True if the demotion completed or was successfully submitted,
            otherwise False.
        """
        cpu_backend = self._get_cpu_backend()
        disk_backend = self._get_disk_backend()
        if cpu_backend is None or disk_backend is None:
            return False

        state = self.hotness_policy.get_state(key)
        if state is None or Tier.CPU not in state.resident_tiers:
            return False

        if hasattr(disk_backend, "contains") and disk_backend.contains(key):
            if self._remove_cpu_if_evictable(cpu_backend, key):
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

        # Use a threading.Event to reliably synchronize callback completion
        # when *blocking* is True, avoiding the race where we read
        # ``demote_result`` before the callback has written it.
        done_event = Event()
        demote_result = [False]  # mutable container for callback to write to

        def _complete_demote(completed_key: CacheEngineKey) -> None:
            try:
                self.hotness_policy.mark_resident(completed_key, Tier.DISK, True)
                if self._remove_cpu_if_evictable(
                    cpu_backend, completed_key
                ) or not cpu_backend.contains(completed_key):
                    self.hotness_policy.mark_resident(
                        completed_key, Tier.CPU, False
                    )
                    demote_result[0] = True
            finally:
                memory_obj.ref_count_down()
                done_event.set()

        put_future = disk_backend.submit_put_task(
            key,
            memory_obj,
            on_complete_callback=_complete_demote,
        )
        if put_future is None:
            memory_obj.ref_count_down()
            return False

        if blocking:
            try:
                put_future.result()
            except Exception:
                logger.exception("Blocking demote failed for key %s", key)
                # Only release if the callback hasn't already run.
                if not done_event.is_set():
                    memory_obj.ref_count_down()
                return False
            # Wait for the callback to finish (may already be done).
            done_event.wait()
            return demote_result[0]

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
        return disk_score > cpu_floor_score + PROMOTION_MARGIN

    def _load_disk_memory_obj(
        self,
        disk_backend: object,
        key: CacheEngineKey,
    ) -> Optional[object]:
        memory_obj = disk_backend.get_blocking(key)
        if memory_obj is None:
            self.hotness_policy.mark_resident(key, Tier.DISK, False)
            return None
        return memory_obj

    def _store_loaded_key_in_cpu(
        self,
        cpu_backend: object,
        key: CacheEngineKey,
        memory_obj: object,
    ) -> bool:
        already_in_cpu = cpu_backend.contains(key)
        cpu_backend.submit_put_task(key, memory_obj)
        promoted = already_in_cpu or cpu_backend.contains(key)
        if promoted and not already_in_cpu:
            self.hotness_policy.mark_resident(key, Tier.CPU, True)

        self._release_memory_obj(memory_obj)
        return promoted

    def _release_memory_obj(self, memory_obj: object) -> None:
        if hasattr(memory_obj, "ref_count_down"):
            memory_obj.ref_count_down()

    def _get_cpu_backend(self) -> Optional[object]:
        if self._cached_cpu_backend is not None:
            return self._cached_cpu_backend
        backend = self.storage_manager.storage_backends.get("LocalCPUBackend")
        if backend is None:
            return None
        required_methods = (
            "get_capacity_bytes",
            "get_usage_bytes",
            "remove",
            "remove_if_evictable",
            "contains",
            "get_blocking",
            "submit_put_task",
        )
        if all(hasattr(backend, method_name) for method_name in required_methods):
            self._cached_cpu_backend = cast(object, backend)
            return self._cached_cpu_backend
        return None

    def _remove_cpu_if_evictable(
        self, cpu_backend: object, key: CacheEngineKey
    ) -> bool:
        if not hasattr(cpu_backend, "remove_if_evictable"):
            return False
        return bool(cpu_backend.remove_if_evictable(key))

    def _get_disk_backend(self) -> Optional[object]:
        if self._cached_disk_backend is not None:
            return self._cached_disk_backend
        backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if backend is None:
            return None
        required_methods = ("get_blocking", "submit_put_task", "contains")
        if all(hasattr(backend, method_name) for method_name in required_methods):
            self._cached_disk_backend = cast(object, backend)
            return self._cached_disk_backend
        return None
