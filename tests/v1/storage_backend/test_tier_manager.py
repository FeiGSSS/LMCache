# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# First Party
from lmcache.v1.storage_backend.hotness_policy import HotnessPolicy, Tier
from lmcache.v1.storage_backend.tier_manager import TierManager
from tests.v1.utils import dumb_cache_engine_key


class FakeMemoryObj:
    def __init__(self) -> None:
        self.ref_count_down_calls = 0

    def ref_count_down(self) -> None:
        self.ref_count_down_calls += 1


class FakeCPUBackend:
    def __init__(self, capacity_bytes: int, usage_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.usage_bytes = usage_bytes
        self.keys: set = set()
        self.removed_keys: list = []
        self.objects = {}

    def get_capacity_bytes(self) -> int:
        return self.capacity_bytes

    def get_usage_bytes(self) -> int:
        return self.usage_bytes

    def remove(self, key) -> bool:
        if key not in self.keys:
            return False
        self.keys.remove(key)
        self.objects.pop(key, None)
        self.removed_keys.append(key)
        self.usage_bytes -= 20
        return True

    def contains(self, key) -> bool:
        return key in self.keys

    def get_blocking(self, key):
        return self.objects.get(key)

    def submit_put_task(self, key, memory_obj) -> None:
        self.keys.add(key)
        self.objects[key] = memory_obj
        self.usage_bytes += 20


class FakeDiskBackend:
    def __init__(self) -> None:
        self.objects = {}

    def get_blocking(self, key):
        return self.objects.get(key)

    def contains(self, key) -> bool:
        return key in self.objects

    def submit_put_task(self, key, memory_obj, on_complete_callback=None):
        self.objects[key] = memory_obj
        if on_complete_callback is not None:
            on_complete_callback(key)
        return object()


def _make_tier_manager(
    hotness_policy: HotnessPolicy,
    cpu_backend: FakeCPUBackend,
    disk_backend: FakeDiskBackend,
) -> TierManager:
    storage_manager = SimpleNamespace(
        storage_backends={
            "LocalCPUBackend": cpu_backend,
            "LocalDiskBackend": disk_backend,
        }
    )
    return TierManager(storage_manager, hotness_policy)


def test_evict_cpu_until_below_watermark_removes_cold_disk_backed_key() -> None:
    hotness_policy = HotnessPolicy()
    cpu_backend = FakeCPUBackend(capacity_bytes=100, usage_bytes=95)
    disk_backend = FakeDiskBackend()
    manager = _make_tier_manager(hotness_policy, cpu_backend, disk_backend)

    cold_key = dumb_cache_engine_key(200)
    hot_key = dumb_cache_engine_key(201)

    hotness_policy.observe_store(cold_key, prefix_pos=8, now=0.0)
    hotness_policy.observe_store(hot_key, prefix_pos=0, now=0.0)
    hotness_policy.on_hit(hot_key, now=0.0)
    hotness_policy.mark_resident(cold_key, Tier.CPU, True)
    hotness_policy.mark_resident(cold_key, Tier.DISK, True)
    hotness_policy.mark_resident(hot_key, Tier.CPU, True)
    hotness_policy.mark_resident(hot_key, Tier.DISK, True)

    cpu_backend.keys.update({cold_key, hot_key})
    cpu_backend.objects[cold_key] = FakeMemoryObj()
    cpu_backend.objects[hot_key] = FakeMemoryObj()

    manager.evict_cpu_until_below_watermark()

    assert cpu_backend.removed_keys == [cold_key]
    assert Tier.CPU not in hotness_policy.get_state(cold_key).resident_tiers


def test_maybe_promote_disk_promotes_hot_disk_only_key() -> None:
    hotness_policy = HotnessPolicy()
    cpu_backend = FakeCPUBackend(capacity_bytes=100, usage_bytes=40)
    disk_backend = FakeDiskBackend()
    manager = _make_tier_manager(hotness_policy, cpu_backend, disk_backend)

    cpu_key = dumb_cache_engine_key(210)
    disk_key = dumb_cache_engine_key(211)
    disk_obj = FakeMemoryObj()
    disk_backend.objects[disk_key] = disk_obj

    hotness_policy.observe_store(cpu_key, prefix_pos=12, now=0.0)
    hotness_policy.observe_store(disk_key, prefix_pos=0, now=0.0)
    hotness_policy.on_hit(disk_key, now=0.0)
    hotness_policy.on_hit(disk_key, now=0.0)
    hotness_policy.mark_resident(cpu_key, Tier.CPU, True)
    hotness_policy.mark_resident(disk_key, Tier.DISK, True)

    cpu_backend.keys.add(cpu_key)
    cpu_backend.objects[cpu_key] = FakeMemoryObj()

    manager.maybe_promote_disk()

    assert cpu_backend.contains(disk_key)
    assert Tier.CPU in hotness_policy.get_state(disk_key).resident_tiers
    assert disk_obj.ref_count_down_calls == 1


def test_demote_key_persists_cpu_only_key_before_removing_cpu() -> None:
    hotness_policy = HotnessPolicy()
    cpu_backend = FakeCPUBackend(capacity_bytes=100, usage_bytes=60)
    disk_backend = FakeDiskBackend()
    manager = _make_tier_manager(hotness_policy, cpu_backend, disk_backend)

    key = dumb_cache_engine_key(220)
    cpu_obj = FakeMemoryObj()

    hotness_policy.observe_store(key, prefix_pos=6, now=0.0)
    hotness_policy.mark_resident(key, Tier.CPU, True)
    cpu_backend.keys.add(key)
    cpu_backend.objects[key] = cpu_obj

    assert manager.demote_key(key)
    assert key in disk_backend.objects
    assert key not in cpu_backend.keys
    state = hotness_policy.get_state(key)
    assert state is not None
    assert Tier.DISK in state.resident_tiers
    assert Tier.CPU not in state.resident_tiers
    assert cpu_obj.ref_count_down_calls == 1
