# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from dataclasses import dataclass
from math import exp, log1p
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType
from lmcache.v1.storage_backend.hotness_constants import (
    AGE_DECAY,
    AGE_WEIGHT,
    HIT_CAP,
    HIT_WEIGHT,
    PREFIX_DECAY,
    PREFIX_WEIGHT,
)

logger = init_logger(__name__)

HOTNESS_BUCKETS = 256
MAX_PREFIX_POS = HOTNESS_BUCKETS - 1
MAX_AGE_TICKS = HOTNESS_BUCKETS - 1

PREFIX_LUT = [
    exp(-prefix_pos / PREFIX_DECAY) for prefix_pos in range(HOTNESS_BUCKETS)
]
AGE_LUT = [exp(-age_tick / AGE_DECAY) for age_tick in range(HOTNESS_BUCKETS)]
LOG_HIT_CAP = log1p(HIT_CAP)


@dataclass
class HotState:
    """Runtime hotness state tracked per cache key within a backend."""

    prefix_pos: int
    hit_count: int
    last_hit_tick: int
    bucket_id: int


class HotnessCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):
    """
    Hotness-aware cache policy.

    The policy keeps the backend cache mapping unchanged and maintains a
    separate bucketed index for eviction decisions. Hotness combines:

    - prefix position within the request
    - age in background-maintenance ticks
    - hit count since insertion

    Thread safety
    -------------
    This class is **not** internally synchronized.  All calls must be
    serialized by the owning backend's lock (the same lock that protects
    the ``cache_dict``).
    """

    def __init__(self) -> None:
        logger.info("Initializing HotnessCachePolicy")
        self.current_tick = 0
        self.states: dict[KeyType, HotState] = {}
        self.key_to_bucket: dict[KeyType, int] = {}
        self.buckets = [OrderedDict() for _ in range(HOTNESS_BUCKETS)]
        self.pending_prefix_pos: dict[KeyType, int] = {}

    def init_mutable_mapping(self) -> OrderedDict[KeyType, Any]:
        return OrderedDict()

    def record_context(
        self,
        key: KeyType,
        *,
        prefix_pos: int | None = None,
    ) -> None:
        if prefix_pos is None:
            return
        self.pending_prefix_pos[key] = min(max(prefix_pos, 0), MAX_PREFIX_POS)

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        self.pending_prefix_pos.pop(key, None)
        state = self.states.get(key)
        if state is None:
            state = self._create_state(key)
            self.states[key] = state

        state.hit_count += 1
        state.last_hit_tick = self.current_tick
        self._reindex_key(key, state)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        state = self._create_state(key)
        self.states[key] = state
        self._reindex_key(key, state)

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        self.pending_prefix_pos.pop(key, None)
        self.states.pop(key, None)
        self._remove_from_bucket_index(key)

    def periodic_maintenance(
        self,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        self.current_tick += 1

        live_keys = set(cache_dict.keys())
        stale_keys = [key for key in self.states if key not in live_keys]
        for key in stale_keys:
            self.states.pop(key, None)
            self.pending_prefix_pos.pop(key, None)

        self.buckets = [OrderedDict() for _ in range(HOTNESS_BUCKETS)]
        self.key_to_bucket.clear()

        for key in cache_dict:
            state = self.states.get(key)
            if state is None:
                state = self._create_state(key)
                self.states[key] = state
            self._reindex_key(key, state)

    def requires_periodic_maintenance(self) -> bool:
        return True

    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []

        for bucket_id in range(HOTNESS_BUCKETS - 1, -1, -1):
            bucket = self.buckets[bucket_id]
            for key in bucket:
                cache = cache_dict.get(key)
                if cache is None:
                    continue
                if not cache.can_evict:
                    continue
                evict_keys.append(key)
                if len(evict_keys) == num_candidates:
                    return evict_keys

        return evict_keys

    def _create_state(self, key: KeyType) -> HotState:
        prefix_pos = self.pending_prefix_pos.pop(key, 0)
        bucket_id = self._compute_bucket_id(
            prefix_pos=prefix_pos,
            hit_count=0,
            last_hit_tick=self.current_tick,
        )
        return HotState(
            prefix_pos=prefix_pos,
            hit_count=0,
            last_hit_tick=self.current_tick,
            bucket_id=bucket_id,
        )

    def _compute_bucket_id(
        self,
        *,
        prefix_pos: int,
        hit_count: int,
        last_hit_tick: int,
    ) -> int:
        prefix_score = PREFIX_LUT[min(max(prefix_pos, 0), MAX_PREFIX_POS)]
        age_ticks = min(max(self.current_tick - last_hit_tick, 0), MAX_AGE_TICKS)
        age_score = AGE_LUT[age_ticks]
        hit_score = min(log1p(hit_count) / LOG_HIT_CAP, 1.0)
        hot_score = (
            PREFIX_WEIGHT * prefix_score
            + AGE_WEIGHT * age_score
            + HIT_WEIGHT * hit_score
        )
        cold_score = 1.0 - hot_score
        return min(max(int(cold_score * (HOTNESS_BUCKETS - 1)), 0), HOTNESS_BUCKETS - 1)

    def _reindex_key(self, key: KeyType, state: HotState) -> None:
        self._remove_from_bucket_index(key)
        state.bucket_id = self._compute_bucket_id(
            prefix_pos=state.prefix_pos,
            hit_count=state.hit_count,
            last_hit_tick=state.last_hit_tick,
        )
        self.buckets[state.bucket_id][key] = None
        self.key_to_bucket[key] = state.bucket_id

    def _remove_from_bucket_index(self, key: KeyType) -> None:
        bucket_id = self.key_to_bucket.pop(key, None)
        if bucket_id is None:
            return
        self.buckets[bucket_id].pop(key, None)
