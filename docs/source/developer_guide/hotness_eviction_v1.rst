V1 Hotness Tiering
==================

Background
----------

V1 originally relies on backend-local cache policies such as ``LRU``, ``LFU``,
``FIFO``, and ``MRU``. Those policies are useful as local fallback eviction
orders, but they do not provide a shared notion of value across tiers. Once V1
needs proactive demotion and promotion, backend-local ordering is no longer
enough because the system must compare:

* a cold key currently resident in CPU
* a hot key currently resident only on disk

That comparison must come from one global score source instead of two unrelated
backend-local policies.

Goals
-----

This phase introduces a global hotness-driven tier-management path while keeping
existing backend eviction code available as fallback.

The scope is intentionally narrow:

* add a global ``HotnessPolicy`` owned by ``StorageManager``
* add a background ``TierManager`` that performs proactive CPU eviction and
  disk-to-CPU promotion
* keep backend-local cache policies in place as fallback when allocator-driven
  pressure still needs a local victim
* keep the current retrieval model and passive write-back behavior intact

This phase does not include:

* remote-tier scheduling
* user-facing configuration for weights, watermarks, or refresh intervals

Architecture
------------

The architecture separates three concerns:

* ``StorageManager`` owns backends and handles request-path store/get/remove
* ``HotnessPolicy`` owns the global per-key hotness state and residency state
* ``TierManager`` consumes hotness and residency to perform proactive
  ``evict`` and ``promote`` actions

Backend-local policies remain in each backend, but when
``cache_policy=\"HOTNESS\"`` is configured they fall back to ``LRU`` internally.
That preserves existing backend behavior for emergency local eviction while the
global hotness path becomes the primary cross-tier decision mechanism.

The repository still contains the older backend-local
``cache_policy/hotness.py`` implementation and its unit tests. That code is now
best understood as a standalone policy primitive and compatibility artifact. It
is not the primary CPU/Disk hotness implementation used when
``cache_policy=\"HOTNESS\"`` is enabled in V1.

Hotness State
-------------

Each key is tracked globally by ``HotnessPolicy``:

.. code-block:: python

   @dataclass
   class HotnessState:
       prefix_pos: int
       hit_count: int
       insert_ts: float
       last_hit_ts: float
       resident_tiers: set[Tier]

The score combines three signals:

* prefix position
* age since the last hit
* hit count since insertion

The current implementation uses lazy age evaluation:

.. code-block:: text

   prefix_score = exp(-prefix_pos / 16.0)
   age_score = exp(-(now - last_hit_ts) / 32.0)
   hit_score = min(log1p(hit_count) / log1p(32), 1.0)

   score = 0.45 * prefix_score + 0.35 * age_score + 0.20 * hit_score

Ageing is an internal detail of ``HotnessPolicy``. The rest of LMCache only
observes scores and residency snapshots.

Residency Tracking
------------------

``HotnessPolicy`` also tracks where each key currently resides:

* ``Tier.CPU``
* ``Tier.DISK``
* ``Tier.REMOTE``

Residency is event-driven:

* ``StorageManager`` updates residency after successful store/remove actions
* passive write-back updates CPU residency only after the CPU put succeeds
* ``TierManager`` updates residency after successful proactive promotion or
  proactive eviction

``HotnessPolicy`` does not read backend internals directly during normal
operation. ``StorageManager`` can still reconcile residency from backend
snapshots during startup or backend recreation.

Call Flow
---------

Store
~~~~~

On ``batched_put()``:

1. ``StorageManager`` calls ``hotness_policy.observe_store(key, prefix_pos)``
2. each backend executes its normal put path
3. on successful persistence, ``StorageManager`` marks the key resident in the
   corresponding tier

Get / Hit
~~~~~~~~~

On ``get()`` or ``batched_get()``:

1. a backend returns a ``MemoryObj``
2. ``StorageManager`` calls ``hotness_policy.on_hit(key)``
3. if the hit came from disk or another non-CPU backend, V1 keeps its existing
   passive write-back behavior and tries to insert the object into CPU
4. CPU residency is updated only if that write-back succeeds

Tier Management
~~~~~~~~~~~~~~~

``TierManager`` runs in the background and uses the global hotness state:

* if CPU usage exceeds the high watermark, it selects the coldest CPU keys and
  proactively demotes them
* if a cold CPU key already exists on disk, the demotion completes immediately
  by removing the CPU copy
* if a cold CPU key exists only on CPU, ``TierManager`` first submits an async
  disk write and removes the CPU copy only after the disk write completion
  callback succeeds
* if CPU usage is below the high watermark, it compares the hottest disk-only
  keys with the coldest CPU keys and promotes disk keys whose score exceeds the
  CPU floor by a promotion margin

Compatibility
-------------

This design keeps the existing backend cache-policy framework compatible:

* ``LocalCPUBackend`` and ``LocalDiskBackend`` still instantiate a backend-local
  cache policy
* when ``cache_policy=\"HOTNESS\"`` is configured, those backend-local policies
  resolve to ``LRU`` fallback instances
* existing allocator-driven or backend-local eviction paths therefore continue
  to work without depending on the global hotness code path

This means the system temporarily has two layers of eviction logic:

* global hotness-driven tier management as the primary strategy
* backend-local ``LRU`` as fallback under immediate local pressure

That is intentional for phase A because it reduces migration risk.

Testing And Acceptance
----------------------

This phase is accepted when:

* ``StorageManager`` creates and owns a global ``HotnessPolicy`` only when
  ``cache_policy=\"HOTNESS\"`` is enabled
* ``TierManager`` starts and stops with ``StorageManager``
* hotness selection prefers colder CPU keys for proactive eviction
* hotness selection prefers hotter disk-only keys for promotion
* successful proactive actions update residency in ``HotnessPolicy``
* backend-local policies continue to function as fallback logic

Future Work
-----------

Likely follow-up work:

* explicit admission control for disk
* remote-tier integration
* configurable scoring weights and watermark thresholds
