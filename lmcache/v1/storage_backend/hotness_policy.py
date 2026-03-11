# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from enum import Enum, auto
from math import exp, log1p
from threading import RLock
from time import time
from typing import Optional

# First Party
from lmcache.utils import CacheEngineKey

HOTNESS_HIT_CAP = 32
HOTNESS_PREFIX_DECAY = 16.0
HOTNESS_AGE_DECAY_SECS = 32.0

HOTNESS_PREFIX_WEIGHT = 0.45
HOTNESS_AGE_WEIGHT = 0.35
HOTNESS_HIT_WEIGHT = 0.20

HOTNESS_PROMOTION_MARGIN = 0.05


class Tier(Enum):
    CPU = auto()
    DISK = auto()
    REMOTE = auto()


@dataclass
class HotnessState:
    prefix_pos: int
    hit_count: int
    insert_ts: float
    last_hit_ts: float
    resident_tiers: set[Tier] = field(default_factory=set)


class HotnessPolicy:
    """
    Global hotness tracker for cross-tier cache management.
    """

    def __init__(self) -> None:
        self._states: dict[CacheEngineKey, HotnessState] = {}
        self._tier_keys: dict[Tier, set[CacheEngineKey]] = {
            Tier.CPU: set(),
            Tier.DISK: set(),
            Tier.REMOTE: set(),
        }
        self._lock = RLock()

    def observe_store(
        self,
        key: CacheEngineKey,
        prefix_pos: int,
        now: Optional[float] = None,
    ) -> None:
        """
        Record a store event and initialize hotness state if needed.
        """
        timestamp = time() if now is None else now
        with self._lock:
            state = self._states.get(key)
            if state is None:
                self._states[key] = HotnessState(
                    prefix_pos=max(prefix_pos, 0),
                    hit_count=0,
                    insert_ts=timestamp,
                    last_hit_ts=timestamp,
                )
                return
            state.prefix_pos = min(state.prefix_pos, max(prefix_pos, 0))

    def on_hit(
        self,
        key: CacheEngineKey,
        now: Optional[float] = None,
    ) -> None:
        """
        Record a cache hit for ``key``.
        """
        timestamp = time() if now is None else now
        with self._lock:
            state = self._states.get(key)
            if state is None:
                self._states[key] = HotnessState(
                    prefix_pos=0,
                    hit_count=1,
                    insert_ts=timestamp,
                    last_hit_ts=timestamp,
                )
                return
            state.hit_count += 1
            state.last_hit_ts = timestamp

    def mark_resident(
        self,
        key: CacheEngineKey,
        tier: Tier,
        present: bool,
    ) -> None:
        """
        Update tier residency for ``key`` after a successful action.
        """
        with self._lock:
            state = self._states.get(key)
            if state is None:
                now = time()
                state = HotnessState(
                    prefix_pos=0,
                    hit_count=0,
                    insert_ts=now,
                    last_hit_ts=now,
                )
                self._states[key] = state

            if present:
                state.resident_tiers.add(tier)
                self._tier_keys[tier].add(key)
                return

            state.resident_tiers.discard(tier)
            self._tier_keys[tier].discard(key)
            if not state.resident_tiers:
                self._states.pop(key, None)

    def clear_tier(self, tier: Tier) -> None:
        """
        Remove residency for all keys in ``tier``.
        """
        with self._lock:
            keys = list(self._tier_keys[tier])
            for key in keys:
                state = self._states.get(key)
                if state is None:
                    continue
                state.resident_tiers.discard(tier)
                if not state.resident_tiers:
                    self._states.pop(key, None)
            self._tier_keys[tier].clear()

    def get_state(self, key: CacheEngineKey) -> Optional[HotnessState]:
        """
        Get current hotness state for ``key``.
        """
        with self._lock:
            return self._states.get(key)

    def get_score(
        self,
        key: CacheEngineKey,
        now: Optional[float] = None,
    ) -> float:
        """
        Get the current hotness score for ``key``.
        """
        timestamp = time() if now is None else now
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return 0.0
            return self._compute_score(state, timestamp)

    def get_keys_in_tier(self, tier: Tier) -> list[CacheEngineKey]:
        """
        Return a snapshot of keys currently resident in ``tier``.
        """
        with self._lock:
            return list(self._tier_keys[tier])

    def select_coldest(
        self,
        tier: Tier,
        limit: int,
        exclude: Optional[set[CacheEngineKey]] = None,
        now: Optional[float] = None,
    ) -> list[CacheEngineKey]:
        """
        Select the coldest keys currently resident in ``tier``.
        """
        timestamp = time() if now is None else now
        excluded = exclude or set()
        with self._lock:
            candidates = [
                key
                for key in self._tier_keys[tier]
                if key not in excluded and key in self._states
            ]
            candidates.sort(
                key=lambda key: self._compute_score(self._states[key], timestamp)
            )
            return candidates[:limit]

    def select_hottest(
        self,
        tier: Tier,
        limit: int,
        require_absent_in: Optional[Tier] = None,
        exclude: Optional[set[CacheEngineKey]] = None,
        now: Optional[float] = None,
    ) -> list[CacheEngineKey]:
        """
        Select the hottest keys currently resident in ``tier``.
        """
        timestamp = time() if now is None else now
        excluded = exclude or set()
        with self._lock:
            candidates = []
            for key in self._tier_keys[tier]:
                if key in excluded or key not in self._states:
                    continue
                state = self._states[key]
                if (
                    require_absent_in is not None
                    and require_absent_in in state.resident_tiers
                ):
                    continue
                candidates.append(key)
            candidates.sort(
                key=lambda key: self._compute_score(self._states[key], timestamp),
                reverse=True,
            )
            return candidates[:limit]

    def refresh(self) -> None:
        """
        Refresh internal aging state.

        Hotness currently uses lazy age computation, so this is a no-op.
        """
        return None

    def _compute_score(self, state: HotnessState, now: float) -> float:
        prefix_score = exp(-state.prefix_pos / HOTNESS_PREFIX_DECAY)
        age_secs = max(now - state.last_hit_ts, 0.0)
        age_score = exp(-age_secs / HOTNESS_AGE_DECAY_SECS)
        hit_score = min(log1p(state.hit_count) / log1p(HOTNESS_HIT_CAP), 1.0)
        return (
            HOTNESS_PREFIX_WEIGHT * prefix_score
            + HOTNESS_AGE_WEIGHT * age_score
            + HOTNESS_HIT_WEIGHT * hit_score
        )
