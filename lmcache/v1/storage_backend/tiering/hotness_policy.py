# SPDX-License-Identifier: Apache-2.0
# Standard
import copy
import heapq
from dataclasses import dataclass, field
from enum import EnumType
from math import exp, log1p
from threading import RLock
from time import time
from typing import Hashable, Optional

# First Party
from lmcache.utils import CacheEngineKey

HIT_CAP = 32
PREFIX_DECAY = 16.0
AGE_DECAY = 32.0

PREFIX_WEIGHT = 0.45
AGE_WEIGHT = 0.35
HIT_WEIGHT = 0.20


@dataclass
class HotnessState:
    prefix_pos: int
    hit_count: int
    insert_ts: float
    last_hit_ts: float
    resident_tiers: set = field(default_factory=set)


class HotnessPolicy:
    """
    Global hotness tracker for cross-tier cache management.

    Tier values are opaque hashable keys — the policy does not depend on
    any specific Tier enum.
    """

    def __init__(self, tier_enum: EnumType) -> None:
        self._states: dict[CacheEngineKey, HotnessState] = {}
        self._tier_keys: dict[Hashable, set[CacheEngineKey]] = {
            t: set() for t in tier_enum
        }
        self._lock = RLock()

    def _get_or_create_state(
        self,
        key: CacheEngineKey,
        timestamp: float,
        prefix_pos: int = 0,
    ) -> HotnessState:
        """
        Return the existing state for *key*, or create and register a new one.

        Must be called while holding ``_lock``.
        """
        state = self._states.get(key)
        if state is None:
            state = HotnessState(
                prefix_pos=max(prefix_pos, 0),
                hit_count=0,
                insert_ts=timestamp,
                last_hit_ts=timestamp,
            )
            self._states[key] = state
        return state

    def observe_put(
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
                self._get_or_create_state(key, timestamp, prefix_pos=prefix_pos)
                return
            state.prefix_pos = min(state.prefix_pos, max(prefix_pos, 0))
            state.last_hit_ts = timestamp

    def observe_get(
        self,
        key: CacheEngineKey,
        now: Optional[float] = None,
    ) -> None:
        """
        Record a cache hit for ``key``.
        """
        timestamp = time() if now is None else now
        with self._lock:
            state = self._get_or_create_state(key, timestamp)
            state.hit_count += 1
            state.last_hit_ts = timestamp

    def add_resident(
        self,
        key: CacheEngineKey,
        tier: Hashable,
    ) -> None:
        """
        Mark ``key`` as resident in ``tier``.

        Raises ``KeyError`` if *key* has no hotness state — callers must
        ensure ``observe_put`` was called first.
        """
        with self._lock:
            state = self._states.get(key)
            if state is None:
                raise KeyError(
                    f"add_resident called for unknown key {key}"
                )
            state.resident_tiers.add(tier)
            self._tier_keys[tier].add(key)

    def remove_resident(
        self,
        key: CacheEngineKey,
        tier: Hashable,
    ) -> None:
        """
        Remove ``key`` from ``tier``. If ``key`` has no remaining tiers,
        its hotness state is deleted.

        No-op if *key* has no hotness state.
        """
        with self._lock:
            state = self._states.get(key)
            if state is None:
                self._tier_keys[tier].discard(key)
                return
            state.resident_tiers.discard(tier)
            self._tier_keys[tier].discard(key)
            if not state.resident_tiers:
                self._states.pop(key, None)

    def clear_tier(self, tier: Hashable) -> None:
        """
        Remove residency for all keys in ``tier``.
        """
        with self._lock:
            for key in self._tier_keys[tier]:
                state = self._states[key]
                state.resident_tiers.discard(tier)
                if not state.resident_tiers:
                    self._states.pop(key, None)
            self._tier_keys[tier].clear()

    def get_state(self, key: CacheEngineKey) -> Optional[HotnessState]:
        """
        Get a snapshot of the current hotness state for ``key``.

        Returns a copy so that callers can read fields without
        holding the policy lock.  The ``resident_tiers`` set is
        copied so mutations do not affect internal state.
        """
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return None
            snapshot = copy.copy(state)
            snapshot.resident_tiers = set(state.resident_tiers)
            return snapshot

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

    def get_keys_in_tier(self, tier: Hashable) -> list[CacheEngineKey]:
        """
        Return a snapshot of keys currently resident in ``tier``.
        """
        with self._lock:
            return list(self._tier_keys[tier])

    def select_coldest(
        self,
        tier: Hashable,
        limit: int,
        exclude: Optional[set[CacheEngineKey]] = None,
        now: Optional[float] = None,
    ) -> list[tuple[CacheEngineKey, float]]:
        """
        Select the coldest keys currently resident in ``tier``.

        Returns a list of ``(key, score)`` pairs sorted coldest-first.
        Uses ``heapq.nsmallest`` — O(n log limit) instead of O(n log n).
        """
        timestamp = time() if now is None else now
        excluded = exclude or set()
        with self._lock:
            scored = [
                (key, self._compute_score(self._states[key], timestamp))
                for key in self._tier_keys[tier]
                if key not in excluded and key in self._states
            ]
            return heapq.nsmallest(limit, scored, key=lambda p: p[1])

    def select_hottest(
        self,
        tier: Hashable,
        limit: int,
        require_absent_in: Optional[Hashable] = None,
        exclude: Optional[set[CacheEngineKey]] = None,
        now: Optional[float] = None,
    ) -> list[tuple[CacheEngineKey, float]]:
        """
        Select the hottest keys currently resident in ``tier``.

        Returns a list of ``(key, score)`` pairs sorted hottest-first.
        Uses ``heapq.nlargest`` — O(n log limit) instead of O(n log n).
        """
        timestamp = time() if now is None else now
        excluded = exclude or set()
        with self._lock:
            scored = []
            for key in self._tier_keys[tier]:
                if key in excluded or key not in self._states:
                    continue
                state = self._states[key]
                if (
                    require_absent_in is not None
                    and require_absent_in in state.resident_tiers
                ):
                    continue
                scored.append(
                    (key, self._compute_score(state, timestamp))
                )
            return heapq.nlargest(limit, scored, key=lambda p: p[1])

    def _compute_score(self, state: HotnessState, now: float) -> float:
        prefix_score = exp(-state.prefix_pos / PREFIX_DECAY)
        age_secs = max(now - state.last_hit_ts, 0.0)
        age_score = exp(-age_secs / AGE_DECAY)
        hit_score = min(log1p(state.hit_count) / log1p(HIT_CAP), 1.0)
        return (
            PREFIX_WEIGHT * prefix_score
            + AGE_WEIGHT * age_score
            + HIT_WEIGHT * hit_score
        )
