# Hotness 分层调度机制说明（中文）

本文档是对当前 V1 hotness 分层调度实现的中文说明，便于内部沟通、设计讨论和给他人讲解机制。

对应英文开发文档：

- `docs/source/developer_guide/hotness_eviction_v1.rst`

相关实现代码：

- `lmcache/v1/storage_backend/tiering/tier_manager.py`
- `lmcache/v1/storage_backend/tiering/hotness_policy.py`
- `lmcache/v1/storage_backend/storage_manager.py`
- `lmcache/v1/storage_backend/local_cpu_backend.py`
- `lmcache/v1/storage_backend/local_disk_backend.py`

## 1. 一句话概括

当前这套 hotness 机制的目标是：

- 给每个 cache key 维护一个全局热度分数
- 跟踪这个 key 当前驻留在哪些层
- 在 CPU 和 Disk 两层之间，根据热度做跨层迁移

更具体地说：

- CPU 压力大时，把冷 key 从 CPU 下沉到 Disk
- 后台线程周期性检查，把更热的 Disk key 提升回 CPU

它是一套“全局打分 + 分层迁移”的机制，而不是简单依赖各 backend 自己的本地 LRU。

## 2. 当前覆盖范围

当前 V1 hotness 路径主要覆盖：

- 全局 hotness 追踪
- CPU / Disk 驻留状态追踪
- CPU 压力触发的即时下沉
- Disk -> CPU 的后台提升
- 从非 CPU 层读取后，仍可被动回写到 CPU

当前还没有覆盖：

- 主动 remote tier 调度
- remote 的跨层晋升 / 下沉策略
- 可配置的复杂打分参数和更细水位线策略
- 通过后台全量扫描来纠正 residency 的 reconcile 机制

## 3. 整体架构

可以把它看成 5 个核心角色：

- `StorageManager`
- `HotnessPolicy`
- `TierManager`
- `LocalCPUBackend`
- `LocalDiskBackend`

结构图如下：

```text
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
```

各部分职责如下：

- `StorageManager`
  - 管请求路径上的 put/get/remove
  - 创建并连接各 backend
  - 在合适时机把事件通知给 hotness 体系
- `HotnessPolicy`
  - 维护每个 key 的热度状态和层驻留状态
  - 提供“最热 / 最冷”选择接口
- `TierManager`
  - 做跨层决策
  - 在 CPU 压力大时执行下沉
  - 在后台周期性做 Disk -> CPU 提升
- `LocalCPUBackend`
  - 维护 CPU 热缓存的实际对象
  - 在 allocator 压力下触发 pressure handler
- `LocalDiskBackend`
  - 维护 Disk 层对象
  - 支持磁盘落盘和磁盘读取

## 4. 设计原则

当前实现有三个核心原则：

1. 热度判断是全局的，不是 backend 各自局部判断
2. 跨层迁移由 `TierManager` 决定，具体存储由 backend 执行
3. residency 通过事件回调维护，而不是周期性扫全后端重建

也就是说：

- “谁更热”由全局 hotness 决定
- “具体怎么删 / 怎么放 / 怎么读”由 backend 决定
- “某 key 当前在哪一层”通过 put/remove/evict callback 持续更新

## 5. 核心概念

### 5.1 hotness

`hotness` 是全局热度分数，用于比较不同 key 的冷热程度。

一个 key 越热，越应该待在高层，比如 CPU。

### 5.2 residency

`residency` 表示系统当前认为某个 key 驻留在哪些层。

当前关注的 tier 是：

- `Tier.CPU`
- `Tier.DISK`

也就是说，某个 key 的驻留可能是：

- 只在 CPU
- 只在 Disk
- 同时在 CPU 和 Disk

### 5.3 demotion

`demotion` 表示把一个 key 从 CPU 移走，同时确保它仍然在 Disk 可用。

它的目标是：

- 释放 CPU 空间
- 不丢失该 key 的可恢复副本

### 5.4 promotion

`promotion` 表示把一个已经在 Disk 的 key 提升到 CPU。

在当前实现里，promotion 主要是 replacement-style：

- 要提升一个更热的 disk key
- 就先把一个更冷的 cpu key 换出去

## 6. Hotness 数据模型

`HotnessPolicy` 为每个 key 维护一个状态，大致包括：

- `prefix_pos`
- `hit_count`
- `insert_ts`
- `last_hit_ts`
- `resident_tiers`

这些字段分别表示：

- 该 key 在前缀中的位置
- 被命中多少次
- 插入时间
- 最近一次命中时间
- 当前驻留在哪些层

其中最重要的是两类信号：

- 热度分数相关信号
- 层驻留状态相关信号

## 7. hotness 分数怎么计算

当前打分由三部分组成：

- prefix 位置
- 最近命中时间
- 历史命中次数

英文文档中的当前公式是：

```text
prefix_score = exp(-prefix_pos / 16.0)
age_score    = exp(-(now - last_hit_ts) / 32.0)
hit_score    = min(log1p(hit_count) / log1p(32), 1.0)

score =
    0.45 * prefix_score +
    0.35 * age_score +
    0.20 * hit_score
```

可以直观理解成：

- 越靠前的 prefix chunk 更重要
- 越近期被访问的 key 更热
- 被重复访问越多的 key 更热

所以它不是简单 LRU，而是综合考虑：

- 前缀位置
- 最近访问
- 访问频次

## 8. 热度状态的生命周期

一个 key 的 hotness state 大致按下面流程进入和离开系统：

```text
observe_store(key, prefix_pos)
        |
        v
  创建或更新 hotness state
        |
        v
mark_resident(key, tier, True)
        |
        v
  resident_tiers 增加对应层
        |
        v
on_hit(key)
        |
        v
  更新 last_hit_ts / hit_count
        |
        v
mark_resident(key, tier, False)
        |
        v
  从 resident_tiers 中删去该层
        |
        v
如果 resident_tiers 为空，则整个 key 的状态被清除
```

也就是说：

- hotness state 不是永久元数据
- 当一个 key 在所有跟踪 tier 中都不存在时，它的状态就会被移除

## 9. residency 是怎么维护的

当前 residency 设计是 callback-driven，不靠定时全盘扫描。

系统通过以下几类事件更新 residency：

- put 完成回调
- `remove()` / `clear()`
- `TierManager` 主动 promotion / demotion
- backend 内部 eviction callback
- backend 生命周期操作，比如 close/recreate

这个设计的含义是：

- 如果 residency 错了，通常意味着某个状态更新回调漏了
- 而不是等一个后台 reconcile 线程慢慢修正

## 10. 事件驱动和后台驱动的分工

当前实现是 hybrid 模式：

- CPU 下沉 / 驱逐：事件驱动
- Disk -> CPU 提升：后台 tick 驱动
- residency 更新：callback 驱动

分工原因是：

- CPU 压力释放必须立即发生，否则 allocator 会失败
- promotion 更像 best-effort 优化，可以批量周期性做

## 11. `StorageManager` 怎么接入 hotness 机制

当 `enable_tiering=True` 时：

- `StorageManager` 创建全局 `HotnessPolicy`
- `StorageManager` 创建 `TierManager`
- 注册 CPU pressure handler
- 注册 CPU / Disk internal evict callback
- 启动后台 `TierManager` 线程

当 `enable_tiering=False` 时：

- 不创建全局 `HotnessPolicy`
- 不启动 `TierManager`
- backend 继续使用各自本地策略

这点非常重要：

- hotness 路径和 backend 本地 LRU 不是同一套机制
- 开启 tiering 后，backend 本地策略仍然存在，只是 hotness 负责跨层动作

## 12. `TierManager` 的职责

`TierManager` 当前主要有两类调度路径：

- `ensure_cpu_headroom()`：同步 CPU 压力释放
- `run_once()` / `maybe_replace_promote_disk()`：后台 promotion 检查

关键常量包括：

- tick interval：默认 `1.0s`
- CPU high watermark：`0.90`
- CPU low watermark：`0.80`
- 每次 tick 最多动作数：`8`

对应代码：`lmcache/v1/storage_backend/tiering/tier_manager.py`

## 13. 并发模型

`TierManager` 用两个锁控制并发：

- `_state_lock`：保护线程生命周期
- `_pressure_lock`：串行化 promotion 和 pressure-relief

这里最重要的正确性要求是：

- 后台提升和前台下沉不能同时改同一批 CPU/Disk 驻留状态

因此：

- `run_once()` 会拿 `_pressure_lock`
- `ensure_cpu_headroom()` 也会拿 `_pressure_lock`

这样避免的问题包括：

- 同一 CPU victim 被双重删除
- 一边 promotion，一边另一条路径把 residency 清掉
- 正在选 victim 时，另一条路径也在改相同对象

## 14. CPU 压力释放流程

CPU 压力路径由 `ensure_cpu_headroom()` 实现。

它的大致流程是：

```text
1. 读取 CPU capacity
2. 计算 target = low_watermark * capacity
3. 如果当前 usage 已经低于 target，直接返回
4. 否则反复：
   - 选择当前最冷的 CPU key
   - 尝试将其下沉
   - 直到 usage 降到 low watermark 以下
   - 或者已经无法继续取得进展
```

几个关键点：

- 目标不是只回到 high watermark 以下，而是进一步回落到 low watermark
- 只考虑 CPU resident 的 key
- 如果 key 被 pin 住或不能 evict，可能导致无法释放足够空间
- 这条路径是同步的，目的是在 allocator 压力下立刻缓解 CPU 容量问题

## 15. Disk -> CPU 提升流程

后台线程会周期性检查：

- 当前哪些 disk key 很热
- 当前哪些 cpu key 很冷

然后尝试做 replacement-style promotion：

```text
选一个 hotter disk key
选一个 colder cpu victim
如果 disk_key 比 victim 更热：
    先从 CPU 移除 victim
    再从 Disk 读出 disk_key
    放回 CPU
```

这里的核心思路是：

- 不做“无脑往 CPU 塞更多东西”
- 而是做“用更热的 key 替换更冷的 key”

当前实现里已经移除了过去的 promotion margin 额外阈值，逻辑更直接：

- 只要 disk key 不比候选 cpu key 更冷，就有机会触发替换提升

## 16. 为什么还保留 backend 本地策略

即使开启 `enable_tiering=True`：

- `LocalCPUBackend` 仍保留本地 eviction 行为
- `LocalDiskBackend` 也仍保留自己的本地 fallback eviction

原因是：

- global hotness 负责跨层决策
- backend 仍需要在紧急 allocator 压力下有自救路径

所以当前系统实际上有两层策略：

- 全局层：hotness-based cross-tier movement
- 本地层：backend-local fallback eviction

## 17. 当前行为可以怎么理解

如果要向别人解释，可以把当前行为总结成：

```text
系统会持续跟踪每个 key 的热度和驻留层。
当 CPU 紧张时，会优先把冷 key 从 CPU 下沉到 Disk。
后台线程则会定期检查 Disk 中是否有更热的 key，值得换回 CPU。
因此 CPU 更像是热层，Disk 更像是冷层；真正的决策依据不是 backend 本地 LRU，
而是全局 hotness 分数。
```

## 18. 当前实现最值得强调的点

如果是做设计评审，建议重点强调下面几点：

- hotness 是全局的，不是 backend 各自算一套
- residency 是 callback-driven，不靠后台全量扫描纠错
- CPU 下沉是事件驱动，promotion 是后台驱动
- promotion 和 pressure-relief 串行化，避免并发踩状态
- backend 本地策略仍在，只是退化成 fallback 机制

## 19. 可能的讨论点

当前实现也有一些天然可以继续讨论的方向：

- 是否需要更丰富的 scoring 权重和可配置参数
- 是否需要更明确的 remote tier 调度逻辑
- 是否要增加 residency reconcile 机制防止状态漂移
- 是否要把 CPU / Disk 的 write-back、promotion、demotion 做成更统一的策略框架

## 20. 文件索引

- 英文开发文档：`docs/source/developer_guide/hotness_eviction_v1.rst`
- 中文说明文档：`docs/source/developer_guide/hotness_eviction_v1_zh.md`
- `TierManager`：`lmcache/v1/storage_backend/tiering/tier_manager.py`
- `HotnessPolicy`：`lmcache/v1/storage_backend/tiering/hotness_policy.py`
- `StorageManager`：`lmcache/v1/storage_backend/storage_manager.py`
- CPU backend：`lmcache/v1/storage_backend/local_cpu_backend.py`
- Disk backend：`lmcache/v1/storage_backend/local_disk_backend.py`
