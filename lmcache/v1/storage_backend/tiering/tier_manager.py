# SPDX-License-Identifier: Apache-2.0
# Standard
import logging
import os
from enum import Enum, auto
from functools import partial
from threading import Event, Lock, Thread
from time import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.tiering.hotness_policy import HotnessPolicy

TIERING_LOG_PATH = os.environ.get(
    "LMCACHE_TIERING_LOG", "/tmp/lmcache_tiering.log"
)


def _init_tiering_logger() -> logging.Logger:
    tlog = logging.getLogger("lmcache.tiering.events")
    tlog.handlers.clear()
    tlog.propagate = False
    fh = logging.FileHandler(TIERING_LOG_PATH, mode="w")
    fh.setFormatter(
        logging.Formatter("%(asctime)s\t%(message)s", datefmt="%H:%M:%S")
    )
    tlog.addHandler(fh)
    tlog.setLevel(logging.DEBUG)
    return tlog


tiering_log = _init_tiering_logger()

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class Tier(Enum):
    CPU = auto()
    DISK = auto()

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

        # Backend references (set in setup_backend_hooks).
        self._cpu_backend: "LocalCPUBackend"
        self._disk_backend: "LocalDiskBackend"

        # Tiering event counters
        self._promote_count = 0
        self._demote_count = 0
        self._pressure_count = 0
        self._tick_count = 0
        self._start_ts = time()

    def start(self) -> None:
        """
        Start the background tier-management loop.
        """
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._start_ts = time()
            self._thread = Thread(
                target=self._run_loop,
                name="storage-tier-manager",
                daemon=True,
            )
            self._thread.start()
            tiering_log.info(
                "START\tinterval=%.2fs\tcpu_high=%.2f\tcpu_low=%.2f\t"
                "max_actions=%d",
                self.interval_secs, self.cpu_high_watermark,
                self.cpu_low_watermark, self.max_actions_per_tick,
            )

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
        self._log_summary()
        tiering_log.info("STOP")

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
        if lcb is None or ldb is None:
            raise RuntimeError(
                "TierManager requires LocalCPUBackend and LocalDiskBackend "
                "to be present in StorageManager"
            )
        
        self._cpu_backend = lcb
        self._disk_backend = ldb

        # CPU eviction is handled by the backend's native LRU policy.
        # TierManager only needs evict callbacks to keep hotness tracking
        # consistent, and the background thread for promotion.

        # Setup evict callbacks
        self._cpu_backend.set_internal_evict_callback(
            partial(self.remove_resident, tier=Tier.CPU)
        )
        self._disk_backend.set_internal_evict_callback(
            partial(self.remove_resident, tier=Tier.DISK)
        )

    def teardown_backend_hooks(self, backend_name: Optional[str] = None) -> None:
        if backend_name is None or backend_name == "LocalCPUBackend":
            self._cpu_backend.set_internal_evict_callback(None)
        if backend_name is None or backend_name == "LocalDiskBackend":
            self._disk_backend.set_internal_evict_callback(None)

    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """
        Run one promotion-management iteration: promote hot disk-only
        keys by replacing colder CPU keys.

        No outer lock is needed here — all shared state is protected by
        fine-grained locks in HotnessPolicy (RLock), LocalCPUBackend
        (cpu_lock), and LocalDiskBackend (disk_lock).  TOCTOU races
        (e.g. a victim evicted between selection and action) are handled
        gracefully: ``remove_if_evictable`` returns ``False`` and the
        promotion is simply skipped.
        """
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
                if disk_score <= cpu_score:
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
                self._promote_count += 1
                tiering_log.info(
                    "PROMOTE\t%s\tscore=%.4f\tvictim=%s\tvictim_score=%.4f",
                    disk_key, disk_score, victim_key, cpu_score,
                )
                logger.debug(
                    "Promoted %s (replacing %s)", disk_key, victim_key
                )

    def ensure_cpu_headroom(self) -> bool:
        """
        Demote coldest CPU-resident keys until usage drops below the low
        watermark.  Called by the CPU backend pressure handler.

        No outer lock — same rationale as ``run_once``.
        """
        cpu = self._cpu_backend
        if cpu.capacity_bytes <= 0:
            return False

        self._pressure_count += 1
        usage_before = cpu.usage_bytes
        target = int(cpu.capacity_bytes * self.cpu_low_watermark)
        relieved = False
        demoted_in_round = 0

        candidates = self.hotness_policy.select_coldest(Tier.CPU)
        for key, score in candidates:
            if cpu.usage_bytes <= target:
                break
            if self.demote_key(key):
                relieved = True
                demoted_in_round += 1
                tiering_log.info(
                    "DEMOTE\t%s\tscore=%.4f\tcpu_usage=%.1fMB",
                    key, score, cpu.usage_bytes / 1e6,
                )

        tiering_log.info(
            "PRESSURE\tdemoted=%d\tcpu_before=%.1fMB\tcpu_after=%.1fMB\t"
            "target=%.1fMB\tcapacity=%.1fMB",
            demoted_in_round,
            usage_before / 1e6,
            cpu.usage_bytes / 1e6,
            target / 1e6,
            cpu.capacity_bytes / 1e6,
        )

        return relieved

    def replace_promote_key(
        self,
        disk_key: CacheEngineKey,
        victim_cpu_key: CacheEngineKey,
    ) -> bool:
        """
        Replace a cold CPU-resident key with a hotter disk-only key.
        Evicts the victim first to avoid unnecessary disk IO on failure.
        """
        # 1. Evict victim from CPU first (cheap, no IO)
        if not self._cpu_backend.remove_if_evictable(victim_cpu_key):
            return False
        self.hotness_policy.remove_resident(victim_cpu_key, Tier.CPU)

        # 2. Load promoted key from disk (IO)
        memory_obj = self._disk_backend.get_blocking(disk_key)
        if memory_obj is None:
            self.hotness_policy.remove_resident(disk_key, Tier.DISK)
            return False

        # 3. Store in CPU
        self._cpu_backend.submit_put_task(disk_key, memory_obj)
        promoted = self._cpu_backend.contains(disk_key)
        if promoted:
            self.hotness_policy.add_resident(disk_key, Tier.CPU)
        memory_obj.ref_count_down()
        return promoted

    def demote_key(self, key: CacheEngineKey) -> bool:
        """
        Demote key from CPU: remove CPU copy (disk already has it per
        the disk ⊇ CPU invariant).
        """
        state = self.hotness_policy.get_state(key)
        if state is None or Tier.CPU not in state.resident_tiers:
            logger.error("demote_key: invalid state for key %s: %s", key, state)
            return False
        if Tier.DISK not in state.resident_tiers:
            return False

        if not self._cpu_backend.remove_if_evictable(key):
            return False
        self.hotness_policy.remove_resident(key, Tier.CPU)
        self._demote_count += 1
        return True

    def _log_summary(self) -> None:
        cpu_keys = len(self.hotness_policy.get_keys_in_tier(Tier.CPU))
        disk_keys = len(self.hotness_policy.get_keys_in_tier(Tier.DISK))
        elapsed = time() - self._start_ts
        cpu_usage = getattr(self._cpu_backend, "usage_bytes", 0)
        cpu_cap = getattr(self._cpu_backend, "capacity_bytes", 0)
        tiering_log.info(
            "SUMMARY\tt=%.1fs\ttick=%d\tcpu_keys=%d\tdisk_keys=%d\t"
            "promotes=%d\tdemotes=%d\tpressures=%d\t"
            "cpu_usage=%.1fMB/%.1fMB",
            elapsed, self._tick_count, cpu_keys, disk_keys,
            self._promote_count, self._demote_count, self._pressure_count,
            cpu_usage / 1e6, cpu_cap / 1e6,
        )

    def _run_loop(self) -> None:
        summary_interval = 10  # log summary every N ticks
        while not self._stop_event.is_set():
            try:
                self._tick_count += 1
                self.hotness_policy.tick_clocks()
                self.run_once()
                if self._tick_count % summary_interval == 0:
                    self._log_summary()
            except Exception:
                logger.exception("TierManager iteration failed")
            self._stop_event.wait(self.interval_secs)



