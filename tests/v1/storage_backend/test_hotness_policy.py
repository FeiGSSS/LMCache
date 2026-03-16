# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.storage_backend.hotness_policy import HotnessPolicy
from lmcache.v1.storage_backend.tier_defs import Tier
from tests.v1.utils import dumb_cache_engine_key


def test_select_coldest_prefers_colder_cpu_key() -> None:
    policy = HotnessPolicy()
    cold_key = dumb_cache_engine_key(100)
    hot_key = dumb_cache_engine_key(101)

    policy.observe_store(cold_key, prefix_pos=8, now=0.0)
    policy.observe_store(hot_key, prefix_pos=0, now=0.0)
    policy.mark_resident(cold_key, Tier.CPU, True)
    policy.mark_resident(hot_key, Tier.CPU, True)

    assert policy.select_coldest(Tier.CPU, limit=2, now=0.0) == [
        cold_key,
        hot_key,
    ]


def test_select_hottest_filters_disk_only_keys() -> None:
    policy = HotnessPolicy()
    disk_only_key = dumb_cache_engine_key(110)
    cpu_and_disk_key = dumb_cache_engine_key(111)

    policy.observe_store(disk_only_key, prefix_pos=0, now=0.0)
    policy.observe_store(cpu_and_disk_key, prefix_pos=0, now=0.0)
    policy.on_hit(disk_only_key, now=1.0)
    policy.on_hit(cpu_and_disk_key, now=1.0)

    policy.mark_resident(disk_only_key, Tier.DISK, True)
    policy.mark_resident(cpu_and_disk_key, Tier.DISK, True)
    policy.mark_resident(cpu_and_disk_key, Tier.CPU, True)

    assert policy.select_hottest(
        Tier.DISK,
        limit=2,
        require_absent_in=Tier.CPU,
        now=1.0,
    ) == [disk_only_key]


def test_mark_resident_removes_state_when_last_tier_cleared() -> None:
    policy = HotnessPolicy()
    key = dumb_cache_engine_key(120)

    policy.observe_store(key, prefix_pos=0, now=0.0)
    policy.mark_resident(key, Tier.DISK, True)
    policy.mark_resident(key, Tier.DISK, False)

    assert policy.get_state(key) is None
