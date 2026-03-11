# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.storage_backend.cache_policy import get_cache_policy

# Local
from .utils import dumb_cache_engine_key


class DummyMemoryObj:
    def __init__(self, can_evict: bool = True):
        self.can_evict = can_evict


def test_lru():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lru_with_pin():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key3, key1]


def test_fifo():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key1, key2]


def test_fifo_with_pin():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lfu():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key1, key3]


def test_lfu_with_pin():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key3, key2]


def test_mru():
    policy = get_cache_policy("MRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    # key1 is the most recent, followed by key3.
    assert evict_candidates == [key1, key3], (evict_candidates, [key1, key3])


def test_mru_with_pin():
    policy = get_cache_policy("MRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj(can_evict=False)  # Pinned object
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    # key1 is most recent, followed by key3, but since key3 is pinned, wo go to key2.
    assert evict_candidates == [key1, key2], (evict_candidates, [key1, key2])


def test_hotness_factory():
    policy = get_cache_policy("HOTNESS")
    assert policy.__class__.__name__ == "HotnessCachePolicy"


def test_hotness_prefix_position():
    policy = get_cache_policy("HOTNESS")
    cache_dict = policy.init_mutable_mapping()
    key1 = dumb_cache_engine_key(11)
    key2 = dumb_cache_engine_key(12)
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()

    policy.record_context(key1, prefix_pos=0)
    cache_dict[key1] = obj1
    policy.update_on_put(key1)

    policy.record_context(key2, prefix_pos=8)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key1]


def test_hotness_hit_count():
    policy = get_cache_policy("HOTNESS")
    cache_dict = policy.init_mutable_mapping()
    key1 = dumb_cache_engine_key(21)
    key2 = dumb_cache_engine_key(22)
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()

    policy.record_context(key1, prefix_pos=0)
    cache_dict[key1] = obj1
    policy.update_on_put(key1)

    policy.record_context(key2, prefix_pos=0)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)

    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key1]


def test_hotness_periodic_maintenance():
    policy = get_cache_policy("HOTNESS")
    cache_dict = policy.init_mutable_mapping()
    key1 = dumb_cache_engine_key(31)
    key2 = dumb_cache_engine_key(32)
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()

    policy.record_context(key1, prefix_pos=0)
    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    policy.periodic_maintenance(cache_dict)

    policy.record_context(key2, prefix_pos=0)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key1, key2]


def test_hotness_with_pin():
    policy = get_cache_policy("HOTNESS")
    cache_dict = policy.init_mutable_mapping()
    key1 = dumb_cache_engine_key(41)
    key2 = dumb_cache_engine_key(42)
    key3 = dumb_cache_engine_key(43)
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj(can_evict=False)
    obj3 = DummyMemoryObj()

    policy.record_context(key1, prefix_pos=0)
    cache_dict[key1] = obj1
    policy.update_on_put(key1)

    policy.record_context(key2, prefix_pos=10)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)

    policy.record_context(key3, prefix_pos=5)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key3, key1]
