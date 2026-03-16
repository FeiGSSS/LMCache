# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from functools import partial
from threading import Event, Lock, Thread
from time import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.tiering.hotness_policy import HotnessPolicy

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class Tier(Enum):
    CPU = auto()
    DISK = auto()

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
        interval_secs: Optional[float] = None,
        cpu_high_watermark: float = DEFAULT_CPU_HIGH_WATERMARK,
        cpu_low_watermark: float = DEFAULT_CPU_LOW_WATERMARK,
        max_actions_per_tick: int = DEFAULT_MAX_ACTIONS_PER_TICK,
    ) -> None:
        self.storage_manager = storage_manager
        self.hotness_policy = HotnessPolicy(Tier)
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

        # Backend references (set in setup_backend_hooks).
        self._cpu_backend: "LocalCPUBackend"
        self._disk_backend: "LocalDiskBackend"

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

    def observe_put(self, keys: Sequence[CacheEngineKey]) -> None:
        now = time()
        for prefix_pos, key in enumerate(keys):
            self.hotness_policy.observe_put(key, prefix_pos, now=now)

    def observe_get(self, key: CacheEngineKey) -> None:
        now = time()
        self.hotness_policy.observe_get(key, now=now)

    def add_resident(self, key: CacheEngineKey, tier: Tier) -> None:
        self.hotness_policy.add_resident(key, tier)

    def remove_resident(self, key: CacheEngineKey, tier: Tier) -> None:
        self.hotness_policy.remove_resident(key, tier)

    def clear_tier(self, tier: Tier) -> None:
        self.hotness_policy.clear_tier(tier)

    def clear_tier_by_name(self, backend_name: str) -> None:
        tier = self._backend_name_to_tier(backend_name)
        if tier is not None:
            self.hotness_policy.clear_tier(tier)

    # ------------------------------------------------------------------
    # Backend name ↔ Tier mapping
    # ------------------------------------------------------------------

    _BACKEND_TO_TIER = {
        "LocalCPUBackend": Tier.CPU,
        "LocalDiskBackend": Tier.DISK,
    }

    @classmethod
    def _backend_name_to_tier(cls, backend_name: str) -> Optional[Tier]:
        tier = cls._BACKEND_TO_TIER.get(backend_name)
        if tier is None:
            logger.warning("Unknown backend name for tiering: %s", backend_name)
        return tier

    # ------------------------------------------------------------------
    # Callback factories for StorageManager put/evict flows
    # ------------------------------------------------------------------

    def make_put_complete_callback(
        self, backend_name: str
    ) -> Optional[Callable[[CacheEngineKey], None]]:
        tier = self._backend_name_to_tier(backend_name)
        if tier is None:
            return None

        def _callback(key: CacheEngineKey) -> None:
            self.add_resident(key, tier)

        return _callback

    def remove_resident_by_name(
        self, key: CacheEngineKey, backend_name: str
    ) -> None:
        tier = self._backend_name_to_tier(backend_name)
        if tier is not None:
            self.remove_resident(key, tier)

    # ------------------------------------------------------------------
    # Backend hook setup
    # ------------------------------------------------------------------

    def setup_backend_hooks(self) -> None:
        backends = self.storage_manager.storage_backends
        unsupported = set(backends.keys()) - {"LocalCPUBackend", "LocalDiskBackend"}
        if unsupported:
            raise RuntimeError(
                f"TierManager only supports LocalCPUBackend and "
                f"LocalDiskBackend, found unsupported: {unsupported}"
            )

        lcb = backends.get("LocalCPUBackend")
        ldb = backends.get("LocalDiskBackend")
        if not (isinstance(lcb, LocalCPUBackend) and isinstance(ldb, LocalDiskBackend)):
            raise RuntimeError(
                "TierManager requires LocalCPUBackend and LocalDiskBackend "
                "to be present in StorageManager"
            )
        
        self._cpu_backend = lcb
        self._disk_backend = ldb

        # Setup CPU pressure handler
        self._cpu_backend.set_pressure_handler(
            self.ensure_cpu_headroom, self.cpu_high_watermark
        )

        # Setup evict callbacks
        self._cpu_backend.set_internal_evict_callback(
            partial(self.remove_resident, tier=Tier.CPU)
        )
        self._disk_backend.set_internal_evict_callback(
            partial(self.remove_resident, tier=Tier.DISK)
        )

    def teardown_backend_hooks(self, backend_name: Optional[str] = None) -> None:
        if backend_name is None or backend_name == "LocalCPUBackend":
            self._cpu_backend.set_pressure_handler(None)
            self._cpu_backend.set_internal_evict_callback(None)
        if backend_name is None or backend_name == "LocalDiskBackend":
            self._disk_backend.set_internal_evict_callback(None)

    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """
        Run one promotion-management iteration.
        """
        with self._pressure_lock:
            self._maybe_replace_promote_disk_locked()

    def maybe_replace_promote_disk(self) -> None:
        """
        Promote hot disk-only keys by replacing colder CPU keys.
        """
        with self._pressure_lock:
            self._maybe_replace_promote_disk_locked()

    def _maybe_replace_promote_disk_locked(self) -> None:
        cpu_backend = self._cpu_backend
        if cpu_backend is None:
            return

        now = time()

        disk_candidates = self.hotness_policy.select_hottest(
            Tier.DISK,
            limit=self.max_actions_per_tick,
            require_absent_in=Tier.CPU,
            now=now,
        )
        if not disk_candidates:
            return

        cpu_candidates = self.hotness_policy.select_coldest(
            Tier.CPU,
            limit=self.max_actions_per_tick,
            now=now,
        )
        if not cpu_candidates:
            return

        logger.debug(
            "Promotion candidates: disk_hot=%d cpu_cold=%d",
            len(disk_candidates),
            len(cpu_candidates),
        )

        used_cpu_keys: set[CacheEngineKey] = set()

        for disk_key, disk_score in disk_candidates:
            victim_key: Optional[CacheEngineKey] = None

            # Iterate hottest-first so we replace the warmest eligible
            # CPU key, leaving colder ones for less-hot disk keys.
            for cpu_key, cpu_score in reversed(cpu_candidates):
                if cpu_key in used_cpu_keys:
                    continue
                if disk_score < cpu_score:
                    continue
                # disk ⊇ CPU invariant: skip if disk copy is missing
                # (should not happen under normal operation)
                state = self.hotness_policy.get_state(cpu_key)
                if state is None or Tier.DISK not in state.resident_tiers:
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
        cpu_backend = self._cpu_backend
        disk_backend = self._disk_backend
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
                for key, _score in candidates:
                    if self.demote_key(key):
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
        cpu_backend = self._cpu_backend
        disk_backend = self._disk_backend
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
        cpu_backend = self._cpu_backend
        disk_backend = self._disk_backend
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

        self.hotness_policy.remove_resident(victim_cpu_key, Tier.CPU)
        return self._store_loaded_key_in_cpu(cpu_backend, disk_key, memory_obj)

    def demote_key(self, key: CacheEngineKey) -> bool:
        """
        Demote key from CPU to disk.

        With the disk ⊇ CPU invariant (batched_put writes to both tiers),
        demotion is a pure in-memory operation: just remove the CPU copy.
        """
        cpu_backend = self._cpu_backend

        state = self.hotness_policy.get_state(key)
        if state is None or Tier.CPU not in state.resident_tiers:
            return False

        # Safety: skip if disk doesn't have it (disk-full edge case)
        if Tier.DISK not in state.resident_tiers:
            return False

        if self._remove_cpu_if_evictable(cpu_backend, key):
            self.hotness_policy.remove_resident(key, Tier.CPU)
            return True
        return False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("TierManager iteration failed")
            self._stop_event.wait(self.interval_secs)

    def _load_disk_memory_obj(
        self,
        disk_backend: object,
        key: CacheEngineKey,
    ) -> Optional[object]:
        memory_obj = disk_backend.get_blocking(key)
        if memory_obj is None:
            self.hotness_policy.remove_resident(key, Tier.DISK)
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
            self.hotness_policy.add_resident(key, Tier.CPU)

        self._release_memory_obj(memory_obj)
        return promoted

    def _release_memory_obj(self, memory_obj: object) -> None:
        if hasattr(memory_obj, "ref_count_down"):
            memory_obj.ref_count_down()

    def _remove_cpu_if_evictable(
        self, cpu_backend: object, key: CacheEngineKey
    ) -> bool:
        if not hasattr(cpu_backend, "remove_if_evictable"):
            return False
        return bool(cpu_backend.remove_if_evictable(key))

