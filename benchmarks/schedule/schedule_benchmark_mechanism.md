# Schedule Benchmark 机制说明

本文档用于解释 `benchmarks/schedule/` 下 TTFT benchmark 的完整工作机制，方便向他人介绍它在“用户、会话、请求、调度、指标”几个层面的设计。

相关入口：

- `benchmarks/schedule/tiering_ttft_bench.py`
- `benchmarks/schedule/src/dataset.py`
- `benchmarks/schedule/src/models.py`
- `benchmarks/schedule/src/user_actor.py`
- `benchmarks/schedule/src/queue.py`
- `benchmarks/schedule/src/scheduler.py`
- `benchmarks/schedule/src/request_client.py`

## 1. 一句话概括

这个 benchmark 模拟的是：

- 一批真实用户
- 每个用户有多条私有多轮历史对话
- 用户在请求完成后等待一段随机时间
- 然后选择继续当前会话或切到另一条历史会话
- 新请求进入全局 ready queue
- 调度器按用户权重从 ready queue 中抽样发出请求
- 每次只生成 `1 token`，主要观测 `TTFT`

它不是“单轮请求压测器”，而是一个面向前缀复用、多会话回流、用户级异步到达的 TTFT workload 生成器。

## 2. 设计目标

这个 benchmark 重点解决三个问题：

1. 让请求携带真实可复用历史，而不是随机拼接 prompt
2. 让热点会话和冷会话回流自然出现，形成缓存冷热分层
3. 尽量压缩 decode 成本，让 TTFT 成为主指标

因此默认行为是：

- 数据来自真实多轮对话
- 每次请求只生成 `1 token`
- 只看首 token 返回时间，不追求长 decode 吞吐

## 3. 总体结构图

可以把整个 benchmark 看成下面这张图：

```text
                    +-------------------------+
                    |   Converted dataset     |
                    | ShareGPT-like messages  |
                    +-----------+-------------+
                                |
                                v
                    +-------------------------+
                    |   User construction      |
                    | num_users                |
                    | conversations_per_user   |
                    | tier/weight assignment   |
                    +-----------+-------------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
     +------------------+               +------------------+
     |   User Actor 0   |               |   User Actor N   |
     | sleep (Poisson)  |               | sleep (Poisson)  |
     | continue/revive  |               | continue/revive  |
     | build request    |               | build request    |
     +--------+---------+               +--------+---------+
              |                                   |
              +-----------------+-----------------+
                                |
                                v
                    +-------------------------+
                    |   WeightedReadyQueue    |
                    | ready requests only     |
                    | weight = user weight    |
                    +-----------+-------------+
                                |
                                v
                    +-------------------------+
                    |   BenchmarkScheduler    |
                    | max_parallel control    |
                    | weighted dispatch       |
                    | result aggregation      |
                    +-----------+-------------+
                                |
                                v
                    +-------------------------+
                    |     vLLM OpenAI API     |
                    | /v1/chat/completions    |
                    | max_tokens = 1          |
                    +-----------+-------------+
                                |
                                v
                    +-------------------------+
                    |    Metrics / Report      |
                    | TTFT / cached_tokens     |
                    | tier / reason / bucket   |
                    +-------------------------+
```

## 4. 三层建模

这个 benchmark 的核心是三层建模：

- 用户层
- 用户内部会话层
- 对话推进层

### 4.1 用户层

这里一个 `client` 表示一个用户，不是一条对话。

每个用户有：

- `user_id`
- `tier`
- `weight`
- 多条私有对话
- 当前状态：`idle / sleeping / queued / in_flight / finished`

默认用户档位和权重是：

- `vip`: 权重 `8`
- `active`: 权重 `3`
- `normal`: 权重 `1`

这些权重不表示用户发请求更快，而是表示：

- 当多个请求已经进入 ready queue 后
- 调度器更容易优先抽中高权重用户的请求

### 4.2 用户内部会话层

每个用户拥有多条真实历史对话，而不是一条固定线程。

用户在一次请求结束后，会在自己的对话集合里选择下一条会话：

- 高概率 `continue`
- 低概率 `revive`

当前实现里：

- `continue`：继续上一次活跃会话
- `revive`：从该用户其他尚未耗尽的对话中随机选一条

这会自然产生：

- 当前热点会话被连续访问
- 旧会话偶尔回流
- 同一用户内部存在冷热切换

### 4.3 对话推进层

每条对话都直接来自真实多轮 `Q/A/Q/A/...` 历史。

例如原始对话是：

```text
Q1, A1, Q2, A2, Q3, A3
```

那么 benchmark 发出的请求序列是：

```text
第1次: [Q1]
第2次: [Q1, A1, Q2]
第3次: [Q1, A1, Q2, A2, Q3]
```

可视化如下：

```text
完整对话:   Q1 -- A1 -- Q2 -- A2 -- Q3 -- A3

请求 #1:    [Q1]
请求 #2:    [Q1, A1, Q2]
请求 #3:    [Q1, A1, Q2, A2, Q3]
```

这意味着：

- 每次请求都停在当前 user turn
- assistant 历史被完整复用进下一次请求
- 前缀会随着会话推进不断变长

## 5. 请求生命周期

一个请求从生成到完成，大致经过下面几步：

```text
1. 用户上一次请求结束
2. 用户按指数分布等待一段时间
3. 用户选择 continue 或 revive
4. 从当前会话状态构造 messages
5. 如果 prompt 过长，则直接跳过该会话
6. 否则把请求放入 ready queue
7. 调度器按权重选中该请求
8. 请求发往 vLLM OpenAI 接口
9. 收到首 token，记录 TTFT
10. 请求结束后推进会话游标
11. 用户重新进入下一轮 sleep -> 选会话 -> 入队
```

对应代码位置：

- 用户请求生成：`benchmarks/schedule/src/user_actor.py`
- 请求执行与统计：`benchmarks/schedule/src/scheduler.py`

## 6. 用户 Actor 状态机

单个用户可以理解成一个异步 actor，状态机会在下面几个状态之间切换：

```text
          +-----------+
          |   IDLE    |
          +-----+-----+
                |
                v
          +-----------+
          | SLEEPING  |  <- 指数分布等待
          +-----+-----+
                |
                v
          +-----------+
          |  QUEUED   |  <- 请求已进入 ready queue
          +-----+-----+
                |
                v
          +-----------+
          | IN_FLIGHT |  <- 请求正在执行
          +-----+-----+
                |
        +-------+--------+
        |                |
        v                v
   +-----------+   +-----------+
   |   IDLE    |   | FINISHED  |
   +-----------+   +-----------+
```

关键约束：

- 同一用户任意时刻最多只有一个请求处于 `queued` 或 `in_flight`

这让 benchmark 更像真实交互用户，而不是固定频率打满并发的压测线程。

## 7. Ready Queue 和全局调度

### 7.1 Ready Queue 里放的是什么

ready queue 里放的是“已经被具体构造出来、随时可以发送”的请求，而不是用户。

队列元素包含：

- `request_id`
- `user_id`
- `conversation_id`
- `user_weight`
- `messages`
- `selection_reason`
- `enqueue_ts`

对应结构：`benchmarks/schedule/src/models.py:93`

### 7.2 调度器怎么选请求

调度器不是“先选用户，再让用户生成请求”，而是：

- 用户自己异步把请求放进队列
- 调度器从队列中的 ready requests 里做加权随机抽样

调度逻辑在：

- `benchmarks/schedule/src/queue.py`
- `benchmarks/schedule/src/scheduler.py`

可视化如下：

```text
Ready Queue:

  req_a (user=vip,    weight=8)
  req_b (user=normal, weight=1)
  req_c (user=active, weight=3)
  req_d (user=normal, weight=1)

调度器抽样概率与用户权重相关：

  vip request    -> 更容易被抽中
  active request -> 次之
  normal request -> 最低
```

因此权重作用的位置是：

- 已入队请求之间的竞争

而不是：

- 用户生成下一条请求的速度

## 8. 泊松到达在这里表示什么

`request_rate_per_user` 控制的是：

- 某个用户上一次请求完成后
- 距离下一次请求入队之前的等待时间

实现上用的是指数分布采样：

- `rng.expovariate(request_rate_per_user)`

对应代码：`benchmarks/schedule/src/user_actor.py:56`

含义是：

- 用户思考下一句话的时间
- 用户再次打开旧会话前的等待
- 用户活跃时间的随机抖动

所以这里要区分两件事：

- 到达时间：由泊松过程控制
- 发出优先级：由 ready queue 抽样权重控制

## 9. 为什么只生成 1 token

benchmark 默认：

- `max_tokens = 1`

目的是尽量把观察重点放在：

- prompt prefill
- prefix 复用
- 首 token 返回时间

而不是：

- 长 decode 吞吐
- 采样策略差异
- 输出长度差异

从测量角度看，这样更接近：

- “缓存是否命中”
- “命中后 TTFT 是否改善”

## 10. Prompt 长度保护

为了避免请求超过模型上下文上限，benchmark 在客户端侧做了 prompt 长度校验。

逻辑是：

- 构造好本次 `messages`
- 用 tokenizer 估算 prompt tokens
- 如果超过 `max_model_len - max_tokens`
- 直接把该会话标记为 exhausted
- 不再把它发给服务端

这样做的作用是：

- 避免服务端返回 400
- 不让超长会话持续污染 workload
- 最终报表里会显示 `超长跳过会话数`

## 11. 指标是怎么采的

请求通过 OpenAI 兼容接口发送：

- `/v1/chat/completions`

客户端开启流式模式：

- `stream=True`
- `stream_options={"include_usage": True}`

对应代码：`benchmarks/schedule/src/request_client.py`

其中：

- `TTFT`：从发请求到收到第一个 `delta.content` 的时间
- `latency_ms`：到 `[DONE]` 为止的总请求时延
- `prompt_tokens`：从 usage 读取
- `cached_tokens`：从 `usage.prompt_tokens_details.cached_tokens` 读取

因此服务端必须带：

```bash
--enable-prompt-tokens-details
```

否则返回里不会有 `cached_tokens`。

## 12. 输出结果怎么看

终端汇总主要包括：

- 总请求数 / 成功数 / 失败数
- 吞吐
- 平均 TTFT / P50 / P90 / P99
- 平均 Prompt Tokens
- 平均缓存 Tokens
- 超长跳过会话数

此外还有三组分桶：

- 按用户档位：`vip / active / normal`
- 按会话选择方式：`continue / revive`
- 按 prompt 长度：`<1k / 1k-4k / 4k-8k / >=8k`

所以一份结果通常可以回答：

- 不同用户档位是否受益一致
- `continue` 和 `revive` 哪种更贵
- 长 prompt 是否显著拉高 TTFT
- prefix cache 命中是否真的转化成 TTFT 改善

## 13. 一个最小示例

假设：

- `num_users = 2`
- 每个用户 2 条对话
- `continue_prob = 0.8`
- `max_parallel = 1`

可以把运行过程理解成：

```text
User0:
  conv_a: [Q1 A1 Q2 A2 Q3 A3]
  conv_b: [Q1 A1 Q2 A2 Q3 A3]

User1:
  conv_c: [Q1 A1 Q2 A2 Q3 A3]
  conv_d: [Q1 A1 Q2 A2 Q3 A3]

时间线:

t0  User0 醒来 -> 生成 req(conv_a:[Q1]) -> 入队
t1  User1 醒来 -> 生成 req(conv_c:[Q1]) -> 入队
t2  调度器按权重从 {req_a, req_c} 中选一个发出
t3  该请求结束，对应用户进入下一轮 sleep
t4  用户再次醒来，80% 概率继续原会话，20% 概率切到另一条
t5  新请求重新进入队列，形成持续 workload
```

## 14. 为什么这个 benchmark 适合讲缓存调度

这个 benchmark 比传统单轮压测更适合讲缓存问题，因为它同时具备：

- 用户级异步到达
- 会话级热点延续
- 冷会话回流
- 前缀长度持续增长
- 明确的 TTFT 指标

因此它能够自然暴露：

- 热前缀是否被持续命中
- 冷前缀回流时是否还能恢复
- 长 prompt 的缓存收益是否足够大
- 不同缓存层或不同后端配置是否真的改善 TTFT

## 15. 向别人介绍时可以怎么讲

如果要快速向别人解释，可以直接用下面这段话：

```text
这个 benchmark 不是把一堆独立 prompt 扔给服务端，而是模拟一批真实用户。
每个用户有多条私有历史会话。用户每次请求结束后，会先等待一个随机时间，
然后高概率继续当前会话、低概率切回其他旧会话。新请求会带着真实多轮历史
进入全局 ready queue，调度器再按用户权重从队列里选请求发出。因为每次只生成
1 个 token，所以最终测到的主要就是前缀复用和 prefill 带来的 TTFT 变化。
```

## 16. 当前语义边界

有几个实现语义最好在介绍时说明清楚：

- `revive` 现在表示“切到另一条尚未耗尽的历史会话”，不要求这条会话之前一定启动过
- 同一用户同一时刻最多只有一个请求在队列中或执行中
- 数据集要求对话从 `user` 开始，严格 `user/assistant` 交替
- 只统计客户端可见的 `cached_tokens`，不直接采信服务端 block pool usage

## 17. 相关文件索引

- 运行入口：`benchmarks/schedule/tiering_ttft_bench.py`
- 参数定义：`benchmarks/schedule/src/config.py`
- 数据加载与用户构造：`benchmarks/schedule/src/dataset.py`
- 状态模型：`benchmarks/schedule/src/models.py`
- 用户 actor：`benchmarks/schedule/src/user_actor.py`
- ready queue：`benchmarks/schedule/src/queue.py`
- 全局调度器：`benchmarks/schedule/src/scheduler.py`
- 请求发送与 TTFT 采集：`benchmarks/schedule/src/request_client.py`
- 设计背景：`benchmarks/schedule/tiering_ttft_benchmark_design.md`
