# Schedule Benchmark 建模文档

本文档统一说明 `benchmarks/schedule/` 下 TTFT benchmark 的建模目标、运行语义、核心调度机制，以及当前实现所对应的代码结构。

相关入口：

- `benchmarks/schedule/tiering_ttft_bench.py`
- `benchmarks/schedule/src/config.py`
- `benchmarks/schedule/src/dataset.py`
- `benchmarks/schedule/src/models.py`
- `benchmarks/schedule/src/user_actor.py`
- `benchmarks/schedule/src/queue.py`
- `benchmarks/schedule/src/scheduler.py`
- `benchmarks/schedule/src/request_client.py`
- `benchmarks/schedule/src/metrics.py`

## 1. 一句话概括

这个 benchmark 模拟的是一批真实用户在多条私有历史会话之间异步切换，用户请求先进入全局 ready queue，再由调度器按用户权重抽样发出；每次请求只生成 `1 token`，主要观测前缀复用和缓存层级对 `TTFT` 的影响。

它不是单轮 prompt 压测器，而是一个面向多用户、多会话、前缀回流和冷热分层的 workload 生成器。

## 2. 建模目标

这个 benchmark 重点解决四件事：

1. 让请求携带真实可复用历史，而不是人工拼接 prompt
2. 让热点会话延续、旧会话回流，从而自然形成冷热分层
3. 把请求到达和全局调度拆开建模
4. 尽量压缩 decode 成本，使 `TTFT` 成为主指标

因此默认语义是：

- 数据来自真实多轮 `user/assistant` 对话
- 每个用户拥有多条私有会话
- 请求完成后，用户按泊松过程等待下一次入队
- 用户高概率继续当前会话，低概率切回其他历史会话
- 请求进入 ready queue 后，由调度器按用户权重随机抽样
- 每次只生成 `1 token`

## 3. 总体结构

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

### 4.1 用户层

这里一个 `client` 表示一个用户，而不是一条对话。

每个用户在 `benchmarks/schedule/src/models.py` 中由 `UserState` 表示，包含：

- `user_id`
- 静态档位 `tier`
- 调度权重 `weight`
- 多条私有对话
- 当前状态 `idle / sleeping / queued / in_flight / finished`
- 最近一次活跃会话 `last_conversation_id`

默认用户档位与权重由 `benchmarks/schedule/src/config.py` 控制：

- `vip = 10%`, `weight = 8`
- `active = 20%`, `weight = 3`
- `normal = 70%`, `weight = 1`

这些权重不决定请求何时到达，只决定已经入队的请求在全局调度阶段被抽中的概率。

### 4.2 用户内部会话层

每个用户拥有多条真实历史会话，而不是只维护一条线程。

在 `benchmarks/schedule/src/user_actor.py` 中，用户每次准备下一条请求时，会做一次“继续还是切换”的选择：

- `continue`：若上次活跃会话尚未耗尽，且命中 `continue_prob`，则继续该会话
- `revive`：否则从该用户其他尚未耗尽的会话中随机选一条

这会自然形成：

- 当前热点会话被连续访问
- 同一用户内部存在冷热切换
- 旧会话偶尔回流，形成低层缓存恢复场景

### 4.3 对话推进层

每条对话在 `benchmarks/schedule/src/models.py` 中由 `ConversationState` 表示，保存完整 `Q/A/Q/A/...` 历史和当前推进位置。

假设原始对话为：

```text
Q1, A1, Q2, A2, Q3, A3
```

那么 benchmark 发出的请求序列是：

```text
第1次: [Q1]
第2次: [Q1, A1, Q2]
第3次: [Q1, A1, Q2, A2, Q3]
```

也就是说：

- 每次请求都停在当前 user turn
- assistant 历史会在后续请求中被完整复用
- 同一会话的前缀会随着推进不断变长

这正是多层缓存调度需要感知的前缀冷热信号。

## 5. 数据与用户构造

`benchmarks/schedule/src/dataset.py` 负责三件事：

### 5.1 数据过滤

仅保留满足以下条件的对话：

- 消息格式为 `{"role", "content"}`
- 对话从 `user` 开始
- 严格 `user/assistant` 交替
- `user` 轮数不少于 `min_user_turns`

### 5.2 用户档位分配

`assign_user_profiles()` 按比例生成 `vip / active / normal` 三类用户，并为其绑定静态权重。

### 5.3 会话分配

`build_users()` 会把过滤后的真实对话随机打散，并按 `conversations_per_user` 切片分配给每个用户。当前实现默认同一条对话只分配给一个用户，不重复共享。

## 6. 请求生命周期

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

对应代码：

- 用户请求生成：`benchmarks/schedule/src/user_actor.py`
- 请求执行与状态推进：`benchmarks/schedule/src/scheduler.py`

## 7. 用户 Actor 状态机

单个用户可以看成一个异步 actor，在下面几个状态之间切换：

```text
          +-----------+
          |   IDLE    |
          +-----+-----+
                |
                v
          +-----------+
          | SLEEPING  |
          +-----+-----+
                |
                v
          +-----------+
          |  QUEUED   |
          +-----+-----+
                |
                v
          +-----------+
          | IN_FLIGHT |
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

这让 benchmark 更像真实交互用户，而不是固定并发压测线程。

## 8. Schedule 机制

### 8.1 两层语义：到达与调度

当前实现里，schedule 机制分成两层：

- 到达过程：用户何时把下一条请求放入 ready queue
- 调度过程：系统何时从 ready queue 中取出请求发给服务端

两者是明确解耦的。

### 8.2 到达过程：泊松等待

在 `benchmarks/schedule/src/user_actor.py` 中，用户每次请求完成后，会根据：

- `rng.expovariate(request_rate_per_user)`

采样一个等待时间，然后再去构造下一次请求。

这表示的是：

- 用户思考下一句话的时间
- 用户重新打开旧会话前的等待
- 用户活跃行为的自然时间抖动

它回答的是：

- 下一个请求什么时候进入 ready queue

### 8.3 调度过程：全局权重抽样

在 `benchmarks/schedule/src/queue.py` 中，`WeightedReadyQueue` 保存所有已经 ready 的请求。

队列元素是 `QueuedRequest`，其中最关键的字段包括：

- `request_id`
- `user_id`
- `conversation_id`
- `user_weight`
- `messages`
- `selection_reason`
- `enqueue_ts`

调度器不是“先选用户，再问用户发什么”，而是：

- 用户自己异步地把具体请求放进队列
- 调度器从队列中的 ready requests 里做加权随机抽样

举例：

```text
Ready Queue:

  req_a (user=vip,    weight=8)
  req_b (user=normal, weight=1)
  req_c (user=active, weight=3)
  req_d (user=normal, weight=1)
```

那么 `vip` 请求更容易被抽中，`active` 次之，`normal` 最低，但不是绝对优先级，也不是严格 FIFO。

因此权重作用的位置是：

- 已入队请求之间的竞争

而不是：

- 用户生成下一条请求的速度

### 8.4 并发控制

在 `benchmarks/schedule/src/scheduler.py` 中，全局调度器维护：

- `ready_queue`
- `in_flight`
- `max_parallel`
- `results`
- `stop_event`

只要满足：

- `in_flight < max_parallel`
- ready queue 非空
- `len(results) + in_flight < max_num_requests`

调度器就会继续从 ready queue 中弹出请求并发执行。

这意味着当前 benchmark 的系统负载上限由 `max_parallel` 控制，而 ready 请求之间的先后顺序则由权重抽样控制。

## 9. 请求执行与状态推进

`BenchmarkScheduler._execute_request()` 的核心逻辑是：

1. 记录队列等待时间 `queue_wait_ms`
2. 调用 `RequestClient.send()` 发送请求
3. 读取返回并记录成功/失败、TTFT、latency、prompt tokens、cached tokens
4. 若成功，则推进对应 `ConversationState`
5. 请求结束后，`in_flight -= 1`
6. 若用户仍有可用会话，则重新进入下一轮 actor 生命周期

因此，请求完成不仅产生指标，也会直接改变后续 workload：

- 当前会话可能继续升温
- 用户可能切回另一条旧会话
- 下一个请求的 prompt 会随着推进而变长

## 10. 为什么只生成 1 token

benchmark 默认 `max_tokens = 1`，目的不是测 decode 吞吐，而是尽量把观测重点放在：

- prompt prefill
- prefix 复用
- 首 token 返回时间

从测量语义上看，这更接近“历史前缀是否命中缓存、命中后 TTFT 是否改善”的问题。

## 11. Prompt 长度保护

为了避免请求超过模型上下文上限，`benchmarks/schedule/src/tokenizer_utils.py` 会在客户端侧做 prompt 长度估算。

逻辑是：

- 用 tokenizer 对当前 `messages` 应用 chat template
- 如果 token 数超过 `max_model_len - max_tokens`
- 直接把该会话标记为 exhausted
- 不再发给服务端

最终报表里会记录：

- `skipped_overlong_conversations`

## 12. 指标采集

`benchmarks/schedule/src/request_client.py` 通过 `/v1/chat/completions` 以流式方式发送请求：

- `stream=True`
- `stream_options={"include_usage": True}`

其中：

- `TTFT`：从发请求到收到第一个 `delta.content` 的时间
- `latency_ms`：到 `[DONE]` 为止的总请求时延
- `prompt_tokens`：从 usage 读取
- `cached_tokens`：从 `usage.prompt_tokens_details.cached_tokens` 读取

所以服务端需要启用：

```bash
--enable-prompt-tokens-details
```

否则结果里不会有可用的 `cached_tokens`。

## 13. 输出结果

`benchmarks/schedule/src/metrics.py` 会汇总以下内容：

- 总请求数 / 成功数 / 失败数
- 运行时长与吞吐
- 平均 TTFT、P50、P90、P99
- 平均 Prompt Tokens
- 平均缓存 Tokens
- 超长跳过会话数

此外还会输出三组分桶：

- 按用户档位：`vip / active / normal`
- 按会话选择方式：`continue / revive`
- 按 prompt 长度：`<1k / 1k-4k / 4k-8k / >=8k`

因此一份结果通常可以回答：

- 高权重用户是否更容易获得更好的 TTFT
- `continue` 和 `revive` 哪种更贵
- 长 prompt 是否显著拉高 TTFT
- `cached_tokens` 的提升是否真的转化成 TTFT 改善

## 14. 一个最小示例

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

## 15. 当前实现边界

介绍这套 benchmark 时，最好同时说明下面几个边界：

- `revive` 当前表示“切到另一条尚未耗尽的历史会话”，不要求这条会话此前一定被访问过
- 同一用户同一时刻最多只有一个请求在队列中或执行中
- 数据集要求对话从 `user` 开始，并严格 `user/assistant` 交替
- 当前队列是轻量内存队列，没有更复杂的公平性、aging 或抢占机制
- 当前统计的是客户端可见的 `cached_tokens`，不直接等价于底层 block 使用细节

## 16. 代码索引

- 运行入口：`benchmarks/schedule/tiering_ttft_bench.py`
- 参数定义：`benchmarks/schedule/src/config.py`
- 数据加载与用户构造：`benchmarks/schedule/src/dataset.py`
- 状态模型：`benchmarks/schedule/src/models.py`
- 用户 actor：`benchmarks/schedule/src/user_actor.py`
- ready queue：`benchmarks/schedule/src/queue.py`
- 全局调度器：`benchmarks/schedule/src/scheduler.py`
- 请求发送与 TTFT 采集：`benchmarks/schedule/src/request_client.py`
- 指标聚合：`benchmarks/schedule/src/metrics.py`

## 17. 对外说明时的简版表述

如果需要快速向别人介绍，可以直接用下面这段话：

```text
这个 benchmark 不是把一堆独立 prompt 扔给服务端，而是模拟一批真实用户。
每个用户有多条私有历史会话。用户每次请求结束后，会先等待一个随机时间，
然后高概率继续当前会话、低概率切回其他旧会话。新请求会带着真实多轮历史
进入全局 ready queue，调度器再按用户权重从队列里选请求发出。因为每次只生成
1 个 token，所以最终测到的主要就是前缀复用和 prefill 带来的 TTFT 变化。
```
