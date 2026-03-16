# SPDX-License-Identifier: Apache-2.0

V1 Hotness Tiering
==================

Overview
--------

This document describes the current V1 hotness-based tier-management
architecture in LMCache. It is a design, implementation, and behavior
specification for the path that is enabled when
``enable_tiering=True``.

At a high level, the system does three things:

* tracks a global hotness score for each cache key
* tracks where each key is currently resident across CPU, disk, and remote tiers
* uses that global view to manage cross-tier movement between CPU and disk

The current implementation is intentionally hybrid:

* CPU demotion / eviction is event-driven
* disk-to-CPU promotion is tick-driven
* residency updates are callback-driven
* backend-local eviction policy still exists as a fallback path

Scope
-----

The current V1 hotness path covers:

* global hotness tracking across the storage manager
* CPU and disk residency tracking
* CPU pressure relief by demoting cold CPU keys
* replacement-style promotion from disk to CPU
* passive CPU write-back on reads from non-CPU tiers

The current V1 hotness path does not cover:

* active remote-tier scheduling
* direct remote promotion / demotion policy
* configurable scoring weights or watermarks
* a backend-scan reconcile pass to rebuild residency

Terminology
-----------

The terms used in this document are:

* ``hotness``: the global score used to compare keys across tiers
* ``residency``: the set of tiers in which a key is currently believed to exist
* ``demotion``: removing a key from CPU while ensuring it exists on disk
* ``promotion``: placing a disk-resident key into CPU
* ``replacement promotion``: promoting a hot disk key by evicting a colder CPU key
* ``event-driven``: triggered synchronously by a concrete runtime event
* ``tick-driven``: triggered by the background ``TierManager`` loop

Architecture
------------

The architecture is split into five roles:

* ``StorageManager``: owns backends, request-path operations, and all hotness wiring
* ``HotnessPolicy``: owns per-key score inputs and tier residency
* ``TierManager``: owns cross-tier scheduling decisions
* ``LocalCPUBackend``: owns actual CPU residency, local fallback eviction, and CPU pressure hooks
* ``LocalDiskBackend``: owns actual disk residency and disk-local fallback eviction

The main design principle is:

* hotness decisions are global
* storage actions are backend-specific
* residency updates are pushed by events instead of rebuilt by scans

System Design
-------------

The following component diagram shows the live control flow:

.. code-block:: text

   Request Path / Background Loop
   ------------------------------

       +----------------+
       | StorageManager |
       +----------------+
          |        |
          |        +------------------------------+
          |                                       |
          v                                       v
   +---------------+                      +---------------+
   | HotnessPolicy |<-------------------->|  TierManager  |
   +---------------+   selection/state    +---------------+
          ^                                       |
          |                                       |
          | callbacks / mark_resident             |
          |                                       v
   +----------------+                      +-----------------+
   | LocalCPUBackend|<-------------------->| LocalDiskBackend|
   +----------------+   move / load / put  +-----------------+

Core Behavior Model
-------------------

The hotness path is built around four rules:

1. ``HotnessPolicy`` is the single runtime source of truth for hotness score and
   tier residency.
2. Residency is updated only by explicit actions and backend callbacks.
3. CPU pressure is relieved immediately on relevant events instead of waiting for
   a background tick.
4. Promotion and demotion are serialized so they do not mutate the same CPU/Disk
   keys concurrently.

Data Model
----------

``HotnessPolicy`` tracks one ``HotnessState`` per key:

.. code-block:: python

   @dataclass
   class HotnessState:
       prefix_pos: int
       hit_count: int
       insert_ts: float
       last_hit_ts: float
       resident_tiers: set[Tier]

Residency tiers are:

* ``Tier.CPU``
* ``Tier.DISK``
* ``Tier.REMOTE``

Hotness Score
-------------

The current score is a weighted sum of three signals:

* prefix position
* age since last hit
* hit count

The implementation in ``lmcache/v1/storage_backend/hotness_policy.py`` is:

.. code-block:: text

   prefix_score = exp(-prefix_pos / 16.0)
   age_score    = exp(-(now - last_hit_ts) / 32.0)
   hit_score    = min(log1p(hit_count) / log1p(32), 1.0)

   score =
       0.45 * prefix_score +
       0.35 * age_score +
       0.20 * hit_score

Interpretation:

* earlier-prefix chunks are favored
* recently-hit chunks are favored
* frequently-hit chunks are favored

The promotion threshold also uses a margin:

.. code-block:: text

   disk_score > cpu_score + HOTNESS_PROMOTION_MARGIN

The current promotion margin is ``0.05``.

Hotness State Lifecycle
-----------------------

State enters and leaves the system as follows:

* ``observe_store(key, prefix_pos)`` creates state if missing
* ``on_hit(key)`` creates state lazily if the key was unknown
* ``mark_resident(key, tier, True)`` adds a tier
* ``mark_resident(key, tier, False)`` removes a tier
* if ``resident_tiers`` becomes empty, the key is removed entirely

This means hotness state is not immortal metadata. A key disappears once it is
not resident in any tracked tier.

Residency Model
---------------

The current residency design is strictly callback-driven. There is no periodic
backend-scan reconcile pass.

Residency is updated by:

* successful put completion callbacks from ``StorageManager``
* explicit ``remove()`` and ``clear()`` calls in ``StorageManager``
* proactive promotion / demotion in ``TierManager``
* backend internal-eviction callbacks from CPU and disk backends
* backend lifecycle operations such as ``close_backend()``, ``recreate_backend()``,
  and ``close()``, which clear tier residency explicitly

This is a deliberate design choice. If residency is wrong, the bug should be a
missing callback or missing explicit state transition, not a missing periodic
reconcile.

Event-Driven vs Tick-Driven Work
--------------------------------

The architecture intentionally separates scheduling paths:

CPU demotion / eviction:

* event-driven
* triggered by CPU allocator pressure
* also triggered immediately after CPU admit if usage crosses the high watermark

Disk-to-CPU promotion:

* tick-driven
* executed by the background ``TierManager`` thread
* only performs replacement-style promotion

Hotness aging:

* technically tick-driven because ``TierManager`` calls ``hotness_policy.refresh()``
* currently a no-op because age is computed lazily

Why this split exists:

* demotion is capacity control and must react immediately
* promotion is best-effort optimization and can tolerate batching

StorageManager Integration
--------------------------

When ``enable_tiering=True``:

* ``StorageManager`` creates a global ``HotnessPolicy``
* ``StorageManager`` creates a ``TierManager``
* ``StorageManager`` registers ``TierManager.ensure_cpu_headroom()`` as the CPU
  pressure handler
* ``StorageManager`` registers internal-evict callbacks on backends that support them
* ``StorageManager`` starts the background ``TierManager`` thread

When ``enable_tiering=False`` (the default):

* no global ``HotnessPolicy`` is created
* no ``TierManager`` is started
* local backend policies continue to run normally

Backend Policy Compatibility
----------------------------

When ``enable_tiering=True``, backend-local eviction policies remain
independent of ``HotnessPolicy``.

* allocator-driven local eviction still works
* backend-local key ordering still exists for emergency fallback
* global hotness only decides cross-tier movement

This is important: the global hotness path and the backend-local policy path are
not the same mechanism.

TierManager Design
------------------

``TierManager`` has two public scheduling paths:

* ``run_once()`` / ``maybe_replace_promote_disk()`` for background promotion
* ``ensure_cpu_headroom()`` for synchronous CPU pressure relief

Important constants:

* tick interval: ``1.0`` second by default
* CPU high watermark: ``0.90``
* CPU low watermark: ``0.80``
* max actions per tick: ``8``

Concurrency Model
-----------------

``TierManager`` uses two locks:

* ``_state_lock``: protects thread lifecycle
* ``_pressure_lock``: serializes promotion and pressure-relief mutations

The critical correctness rule is:

* background promotion and event-driven demotion must not operate on CPU/Disk
  residency concurrently

Therefore:

* ``run_once()`` acquires ``_pressure_lock``
* ``maybe_replace_promote_disk()`` acquires ``_pressure_lock``
* ``ensure_cpu_headroom()`` acquires ``_pressure_lock``

This prevents races such as:

* double-evicting the same CPU victim
* clearing CPU residency before a concurrent promotion finishes
* selecting a victim while another path is already removing it

CPU Pressure-Relief Algorithm
-----------------------------

The CPU pressure path is implemented by ``TierManager.ensure_cpu_headroom()``.

Algorithm:

1. read CPU capacity from ``LocalCPUBackend.get_capacity_bytes()``
2. compute ``target_bytes = capacity * low_watermark``
3. if current usage is already below target, do nothing
4. otherwise repeatedly:

   * select the coldest CPU-resident keys
   * demote them one by one
   * stop when usage falls below the low watermark
   * stop early if no progress is possible

Important properties:

* the CPU is drained to the low watermark, not merely back under the high watermark
* only CPU-resident keys are considered
* pinned or otherwise non-evictable keys can block progress
* the path is synchronous and returns whether any headroom was created

Replacement Promotion Algorithm
-------------------------------

The promotion path is implemented by ``TierManager.maybe_replace_promote_disk()``.

Algorithm:

1. select the hottest keys in ``Tier.DISK`` that are absent from ``Tier.CPU``
2. select the coldest keys in ``Tier.CPU``
3. for each hot disk key:

   * find a CPU victim that is colder by at least the promotion margin
   * require that the CPU victim is also resident on disk
   * replace the CPU victim with the disk key

Why the victim must already exist on disk:

* replacement promotion is designed to keep CPU capacity stable
* removing a CPU-only key without first persisting it would cause data loss

Why this is replacement-style rather than free-space promotion:

* the current implementation does not proactively fill idle CPU headroom
* promotion is only attempted by swapping in a hotter disk key

Promotion Safety Guarantees
---------------------------

The current implementation explicitly avoids the earlier promotion bug where the
CPU victim was dropped before the promoted key was proven promotable.

Current replacement flow:

1. load the promoted key from disk first
2. if disk load fails, abort and keep the victim
3. verify the victim still has a disk copy
4. remove the victim from CPU only if it is evictable
5. store the loaded key into CPU
6. update CPU residency only if the store actually succeeded

This ensures that a missing or stale disk candidate does not cause avoidable CPU
cache regression.

Demotion Safety Guarantees
--------------------------

``TierManager.demote_key()`` has two cases:

Case 1: the key already exists on disk

* remove the CPU copy if it is evictable
* mark ``Tier.CPU = False``
* mark ``Tier.DISK = True``

Case 2: the key exists only on CPU

* fetch the CPU memory object
* submit disk put
* on disk put completion:

  * mark ``Tier.DISK = True``
  * remove CPU copy if evictable
  * mark ``Tier.CPU = False`` only after CPU removal succeeds

This ordering preserves the rule that a CPU key must not be forgotten before a
durable disk copy exists.

Capacity Accounting
-------------------

The watermarks operate on the effective CPU capacity, not just the raw config.

``LocalCPUBackend.get_capacity_bytes()`` reflects the actual allocator capacity
after adjustments such as:

* effective CPU size calculation
* reserve CPU memory
* first-rank CPU size override

This matters because ``TierManager`` uses ``get_capacity_bytes()`` as the
denominator for watermark decisions. Using the configured size instead of the
effective size would skew demotion timing.

Hotness Request-Path Flows
--------------------------

Store Path
~~~~~~~~~~

When ``StorageManager.batched_put()`` stores keys:

* it first records ``observe_store()`` for each key with its prefix position
* it submits put operations to all selected backends
* each backend put completion callback marks residency for that backend's tier

Sequence:

.. code-block:: text

   Client
     |
     v
   StorageManager.batched_put(keys, objs)
     |
     +--> HotnessPolicy.observe_store(key, prefix_pos)
     |
     +--> LocalCPUBackend.batched_submit_put_task(..., on_complete=mark CPU resident)
     |
     +--> LocalDiskBackend.batched_submit_put_task(..., on_complete=mark DISK resident)
     |
     +--> RemoteBackend.submit_put_task(..., on_complete=mark REMOTE resident)

Hit / Read Path
~~~~~~~~~~~~~~~

When ``StorageManager.get()`` or ``batched_get()`` reads a key:

* a successful hit records ``on_hit()``
* if the hit comes from a non-CPU backend and CPU exists, the object is
  passively written back into CPU
* CPU residency is marked only after the CPU put completes

Sequence:

.. code-block:: text

   Client
     |
     v
   StorageManager.get(key)
     |
     +--> backend.get_blocking(key)
             |
             +--> hit
     |
     +--> HotnessPolicy.on_hit(key)
     |
     +--> if backend != CPU and LocalCPUBackend exists:
             LocalCPUBackend.submit_put_task(
                 key,
                 memory_obj,
                 on_complete=mark CPU resident,
             )

CPU Pressure Sequence
~~~~~~~~~~~~~~~~~~~~~

There are two event sources for CPU pressure relief:

* allocator failure before local fallback eviction
* successful admit that pushes CPU usage above the high watermark

Sequence:

.. code-block:: text

   LocalCPUBackend.allocate() / submit_put_task()
     |
     +--> pressure handler registered by StorageManager
             |
             v
       TierManager.ensure_cpu_headroom()
             |
             +--> select coldest CPU keys
             +--> demote_key(...)
             +--> loop until CPU usage <= low watermark

Tick Promotion Sequence
~~~~~~~~~~~~~~~~~~~~~~~

The background thread handles promotion only.

Sequence:

.. code-block:: text

   TierManager thread
     |
     v
   run_once()
     |
     +--> acquire _pressure_lock
     +--> HotnessPolicy.refresh()
     +--> select hottest disk-only keys
     +--> select coldest CPU keys
     +--> replace_promote_key(...)
     +--> release _pressure_lock

Internal Eviction Callback Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a backend evicts a key internally under its own local policy:

* the backend invokes its internal-evict callback
* ``StorageManager`` maps the backend name to a tier
* ``HotnessPolicy.mark_resident(key, tier, False)`` runs immediately

Sequence:

.. code-block:: text

   LocalCPUBackend / LocalDiskBackend
     |
     +--> _notify_internal_evict(key)
             |
             v
       StorageManager callback
             |
             v
       HotnessPolicy.mark_resident(key, tier, False)

Lifecycle Behavior
------------------

Startup
~~~~~~~

On ``StorageManager`` creation with ``enable_tiering=True``:

* create backends
* create ``HotnessPolicy``
* create ``TierManager``
* register CPU pressure handler
* register backend internal-evict callbacks
* start ``TierManager``

Backend Close / Recreate
~~~~~~~~~~~~~~~~~~~~~~~~

On backend close or recreation:

* backend pressure handler is removed if needed
* backend internal-evict callback is removed
* backend is closed
* corresponding hotness tier residency is cleared explicitly
* callbacks / pressure handler are refreshed after recreation

Manager Shutdown
~~~~~~~~~~~~~~~~

On ``StorageManager.close()``:

* ``TierManager`` is stopped first
* CPU pressure handler is cleared
* backend callbacks are cleared
* each backend tier residency is cleared
* all backends are closed

Correctness Invariants
----------------------

The current implementation relies on the following invariants:

* ``HotnessPolicy`` is the runtime source of truth for residency
* residency changes only through explicit events and callbacks
* CPU demotion is event-driven
* promotion is tick-driven
* promotion and demotion are serialized by ``_pressure_lock``
* a CPU victim is removed only if it is currently evictable
* a replacement promotion only uses CPU victims that are also present on disk
* a key is removed from hotness state when it is no longer resident anywhere

Failure Semantics
-----------------

If a disk promotion candidate disappears before promotion:

* disk residency is cleared
* CPU victim is kept
* promotion returns ``False``

If the CPU victim is pinned or otherwise non-evictable:

* replacement promotion is skipped
* demotion is skipped
* CPU pressure relief may make no progress

If a demotion disk write fails:

* CPU residency remains
* disk residency is not marked
* the operation returns failure

If a backend is closed or recreated:

* its tier residency is cleared explicitly
* no attempt is made to rebuild it by scanning backend contents

Current Limitations
-------------------

The current design is deliberately conservative.

Known limitations:

* promotion is replacement-only and does not fill free CPU headroom
* remote-tier scheduling is not implemented
* hotness weights, watermarks, and tick interval are not user-configurable
* correctness depends on callback completeness because there is no reconcile scan
* if most CPU keys are not evictable, pressure relief can stall

Implementation Map
------------------

The main implementation files are:

* ``lmcache/v1/storage_backend/hotness_policy.py``
* ``lmcache/v1/storage_backend/tier_manager.py``
* ``lmcache/v1/storage_backend/storage_manager.py``
* ``lmcache/v1/storage_backend/local_cpu_backend.py``
* ``lmcache/v1/storage_backend/local_disk_backend.py``

The main regression and behavior tests are:

* ``tests/v1/storage_backend/test_tier_manager.py``
* ``tests/v1/storage_backend/test_storage_manager.py``
* ``tests/v1/storage_backend/test_local_cpu_backend.py``
* ``tests/v1/test_cache_policy.py``

Functional Summary
------------------

With ``enable_tiering=True``, the system provides:

* global hotness-aware cross-tier decisions
* callback-accurate tier residency tracking
* immediate CPU pressure relief when CPU capacity is exceeded
* safe replacement promotion from disk to CPU
* passive CPU warming on non-CPU reads
* compatibility with existing backend-local fallback eviction

In short, the current design treats hotness as the global policy layer and
keeps backend-local policies as local safety nets.

Future Work
-----------

Natural next steps for this architecture are:

* free-headroom promotion instead of replacement-only promotion
* remote-tier scheduling
* configurable policy parameters
* richer observability for promotion / demotion decisions
* optional audit tooling for verifying callback completeness
