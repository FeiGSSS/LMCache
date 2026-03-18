# 面向多层调度的 TTFT Benchmark 设计方案

## 1. 设计动机

多层级缓存调度系统的最终目标，不是优化 decode 阶段的吞吐，而是降低请求到达时的首 token 返回时间，也就是 TTFT。

从请求生命周期来看，真正能被多层缓存命中的部分，主要是历史上下文对应的前缀 KV，以及当前轮新增输入在 prefill 阶段的处理开销。因此，一个用于评估多层调度效果的 benchmark，应当尽可能满足以下目标：

- 重点考察前缀复用与 prefill 行为
- 重点测量 TTFT，而不是长 decode 吞吐
- 尽量使用真实多轮对话数据中的历史问答内容
- 让工作负载自然形成冷热分层
- 让冷会话能够被重新唤醒，从而检验下层缓存的恢复价值

从系统形态上看，这样的 benchmark 更接近 PD 分离架构中的 Prefill 节点负载：每次请求都携带较长历史，系统主要工作在前缀匹配、缓存命中、补齐少量新增上下文，以及生成首 token。

## 2. 总体目标

本设计希望构建一个新的 TTFT 导向 benchmark，用于评估多层缓存调度在真实多用户、多会话、多轮历史复用场景下的效果。

该 benchmark 的核心目标包括：

- 让每次请求都携带可复用的真实历史前缀
- 让热用户、热会话、冷会话回流都能在负载中自然出现
- 通过最小化 decode 成本，使 TTFT 成为主要观测指标
- 为 CPU / disk 等多层缓存调度提供足够明显的冷热信号

## 3. 负载建模总览

整个 benchmark 采用三层建模方式：

- 用户层
- 用户内部会话层
- 对话推进层

这三层分别描述：谁更活跃、同一用户当前在做哪条会话、这条会话如何沿着真实历史推进。

### 3.1 用户层

一个 `client` 表示一个用户，而不是一条单独对话。

每个用户具备以下属性：

- 静态活跃档位
- 对应的调度权重
- 多条真实历史对话
- 任意时刻最多只有一个处于 `queued` 或 `in_flight` 的请求

静态活跃档位用于描述不同用户整体活跃程度的差异。例如：

- `vip`
- `active`
- `normal`

这些档位不会决定请求何时到达，只决定该用户的请求在全局调度阶段被优先取出的概率。

### 3.2 用户内部会话层

每个用户并不只拥有一条对话，而是拥有多条真实历史会话。这个设定更接近真实订阅用户：同一个用户可能同时有天气、论文、代码、旅游等不同主题的历史对话。

当某个用户上一个请求完成后，用户会决定下一次要继续哪条会话。这个选择分为两种情况：

- 以较高概率继续上一次活跃的会话
- 以较低概率唤醒该用户的另一条历史会话

这样可以同时模拟两类真实行为：

- 用户围绕当前主题连续追问，形成短期高热会话
- 用户偶尔重新打开旧话题，形成冷会话回流

### 3.3 对话推进层

每条对话都直接使用真实数据集中已有的多轮 `Q/A/Q/A/...` 结构，不再重新拼接问题，也不再依赖模型在线生成的回复去构造下一轮历史。

假设一条真实对话为：

- `Q1, A1, Q2, A2, Q3, A3, ...`

那么该 benchmark 在用户边界上发出的请求为：

- 第 1 次请求：`[Q1]`
- 第 2 次请求：`[Q1, A1, Q2]`
- 第 3 次请求：`[Q1, A1, Q2, A2, Q3]`

也就是说，每次请求都停在当前用户输入这一轮，而历史中的 assistant 回复会作为下一轮请求的一部分被完整复用。

这种构造方式有两个重要意义：

- 真正使用了数据中通常更长的 assistant 回复内容，能更真实地形成大前缀
- 每次请求都更贴近“当前用户发来新问题，系统进行 prefill 并返回首 token”的实际链路

## 4. 为什么这个负载适合多层调度

多层调度依赖的是显著的冷热差异，而这种差异会同时来自三个维度：

- 用户级冷热：VIP 用户整体更活跃，请求更多、更频繁被调度
- 会话级冷热：同一用户更倾向于连续推进当前会话
- 前缀级冷热：被频繁访问的长历史前缀会持续出现，而较老的历史会话则可能沉到底层，之后再被唤醒

因此，这个 benchmark 可以自然产生以下现象：

- 热用户的若干热点会话不断命中上层缓存
- 长尾用户和冷门会话更多依赖下层存储
- 被重新唤醒的旧会话会形成“从低层恢复”的典型访问路径

这些行为正好对应多层缓存调度最想优化的对象：

- 热前缀尽量留在更高层
- 冷前缀可以沉到底层
- 被重新访问的冷前缀应尽快恢复并改善 TTFT

## 5. 为什么仍然需要泊松到达

在这个设计中，泊松分布不是用来表示用户优先级，而是用来模拟“一个用户在上一个请求完成之后，下一次请求何时到来”。

它对应的是：

- 用户思考下一句话所花的时间
- 用户切换到其他历史会话之前的等待时间
- 用户再次活跃的自然时间抖动

也就是说，泊松建模回答的是：

- 下一个请求什么时候进入全局队列

而静态用户权重回答的是：

- 进入全局队列之后，这个请求被优先发出的概率有多高

这两者应当严格分离：

- 到达时间描述请求生成过程
- 权重描述系统调度过程

这样构造后，负载会更接近真实线上环境：请求不是在固定节拍下同步产生的，而是在用户交互完成后，以随机间隔逐步进入系统。

## 6. 运行时语义

整个 benchmark 应采用异步运行方式。

核心上包含两类实体：

- 每个用户一个异步 actor
- 一个全局异步调度器

### 6.1 用户 actor 的职责

每个用户 actor 负责维护该用户的全部状态，包括：

- 拥有哪些真实会话
- 上一次活跃的是哪条会话
- 当前是否已有请求排队或在飞
- 上一个请求完成后如何生成下一次请求

用户 actor 的基本生命周期如下：

1. 某次请求被发出
2. 等待该请求返回
3. 请求返回后，记录本次 TTFT
4. 根据泊松分布采样下一次请求的等待时间
5. 异步休眠该时间
6. 休眠结束后，在用户自己的历史会话中做一次“继续 / 唤醒”选择
7. 构建新请求并放入全局 ready queue

一个关键约束是：

- 同一用户在任意时刻最多只允许有一个请求处于 `queued` 或 `in_flight`

这个约束保证 benchmark 更接近真实交互：一个用户提交请求后，需要等服务端回复，才能继续发出下一次请求。

### 6.2 全局调度器的职责

全局调度器维护以下全局状态：

- ready queue
- 当前 in-flight 请求数
- 最大并发 `max_parallel`
- 请求与所属用户、所属会话的映射
- 聚合统计指标

当系统存在并发空位，即 `in_flight < max_parallel` 时，全局调度器就从 ready queue 中选择一个请求发往服务端。

选择时采用按权重随机抽样：

- 抽样对象是“已经入队的请求”
- 抽样权重取自该请求所属用户的静态权重

请求完成后：

- in-flight 计数减一
- 将结果写入统计
- 通知所属用户 actor 进入下一轮泊松等待和请求规划流程

## 7. 请求是如何进入队列和被调度的

该 benchmark 的核心调度语义是：

- 用户自己异步地产生请求
- 请求进入全局队列
- 全局调度器从队列中按用户权重抽样并发出

也就是说，不是“先选用户，再问用户发什么”，而是：

1. 某个用户上一次请求结束
2. 该用户等待一个泊松采样得到的时间
3. 该用户选择下一条会话
4. 该用户构造一个具体请求并放入 ready queue
5. 全局调度器在所有 ready 请求中，按所属用户权重做加权选择

这样做有两个重要好处：

- 到达过程和调度过程被清晰解耦
- 用户优先级作用在系统真实排队面上，而不是作用在“想象中的下一步用户选择”上

## 8. 数据模型设计

### 8.1 数据集要求

数据集需要提供真实多轮对话，并且具备清晰的用户 / assistant 交替结构。

每条原始对话应被规范化为：

- `conversation_id`
- 有序 turn 列表：`[(user, Q1), (assistant, A1), (user, Q2), (assistant, A2), ...]`

可用于 benchmark 的对话至少应满足：

- 包含第一轮用户问题
- 包含至少一轮 assistant 回复，用于后续历史复用
- 最好包含至少两个用户轮次，以便同一会话可以发起多次请求

### 8.2 用户对象

每个用户应持有多条从数据集中抽取的真实对话。

例如：

- `num_users = 128`
- `conversations_per_user = 4`

同一条对话默认不应重复分配给多个用户，除非后续实验有意引入重复副本。

### 8.3 会话状态

每条会话需要维护以下信息：

- 会话 ID
- 全部 turn 列表
- 当前推进到哪一个用户 turn
- 是否已耗尽
- 最近一次被访问的时间
- 被复用次数

对于当前推进到 `Qt` 的会话，请求内容由从开头到 `Qt` 的全部历史构成，不包含未来轮次。

例如：

- 当前在 `Q1`：请求为 `[Q1]`
- 当前在 `Q2`：请求为 `[Q1, A1, Q2]`
- 当前在 `Q3`：请求为 `[Q1, A1, Q2, A2, Q3]`

每次请求结束后，会话推进到下一个用户 turn，等待未来再次被选中时继续发送。

## 9. 用户内部的会话选择策略

当某个用户准备生成下一次请求时，会进行两阶段选择。

### 9.1 第一阶段：继续还是唤醒

设定一个参数：

- `p_continue_same_conversation`

语义如下：

- 如果上一次活跃会话尚未结束，并且一次伯努利采样命中，则继续该会话
- 否则，从该用户的其他可用历史会话中选择一条进行唤醒

推荐第一版默认值：

- `p_continue_same_conversation = 0.8`

### 9.2 第二阶段：唤醒哪一条历史会话

第一版中，唤醒策略建议保持简单：

- 在当前用户其他仍可继续推进的会话中做均匀随机选择

后续可扩展为：

- 基于最近访问时间的偏置
- 基于历史冷热程度的偏置
- 基于对话长度的偏置

## 10. 用户活跃档位与权重

每个用户在 benchmark 初始化时被赋予一个静态档位。

推荐第一版采用三档：

- `vip`
- `active`
- `normal`

推荐占比：

- `vip = 10%`
- `active = 20%`
- `normal = 70%`

推荐权重：

- `vip = 8`
- `active = 3`
- `normal = 1`

这些权重只在全局调度器从 ready queue 中选请求时生效，不直接影响用户内部会话选择，也不直接影响泊松等待时间。

## 11. 到达时间模型

### 11.1 基础模型

当某个用户的一次请求完成后，该用户不会立即把下一次请求放入队列，而是先采样一个等待时间，再异步休眠，然后再提交下一次请求。

推荐第一版使用指数分布：

- `delay ~ Exp(lambda = request_rate_per_user)`

也就是每个用户都独立拥有自己的请求到达过程。

这种设计意味着：

- 短等待更常见
- 长等待偶尔发生
- 不同用户不会在固定周期上同步发请求

### 11.2 为什么按用户采样更合理

这个 benchmark 的建模对象是用户，而不是全局固定节拍请求流。因此，请求到达应当是“每个用户在自己的交互节奏上独立演化”，而不是由全局统一时钟批量生成。

这也更符合用户 actor 的设计：

- 请求完成
- 用户思考
- 用户继续当前会话或唤醒旧会话
- 下一次请求入队

## 12. TTFT-only 测量策略

该 benchmark 的重点是 TTFT，因此请求应尽可能压低 decode 成本。

推荐配置：

- `max_new_tokens = 1`
- 尽量使用确定性解码
- 尽量减少不必要的采样波动

每条请求建议记录如下信息：

- TTFT
- prompt token 长度
- cached tokens 数量（如果后端可提供）
- 用户档位
- 用户 ID
- 会话 ID
- 本次是 continue 还是 revive
- 入队时间
- 开始发送时间
- 完成时间
- 队列等待时间

建议输出的聚合指标包括：

- 平均 TTFT
- p50 / p90 / p99 TTFT
- 按用户档位分组的 TTFT
- 按 prompt 长度分桶的 TTFT
- 按 continue / revive 分组的 TTFT
- cache hit rate（如果可观测）

如果后续能从服务端额外拿到更细粒度的层级缓存信息，还可以增加：

- CPU 命中次数
- disk 命中次数
- promotion 次数
- demotion 次数
- 冷会话恢复次数

## 13. 停止条件

benchmark 应支持如下停止方式之一：

- 达到最大请求数 `max_num_requests`
- 达到指定运行时长 `duration_s`

推荐第一版先支持：

- `max_num_requests`

当达到停止条件时：

- 不再允许新的请求入队
- 已经 in-flight 的请求继续执行到结束
- 最后统一汇总统计指标

## 14. 建议暴露的参数

### 14.1 数据与请求构造

- `--dataset-path`
- `--num-users`
- `--conversations-per-user`
- `--min-turns-per-conversation`
- `--max-turns-per-conversation`
- `--max-new-tokens`
- `--api-format`
- `--model-path`
- `--host`
- `--port`

### 14.2 用户档位与权重

- `--vip-ratio`
- `--active-ratio`
- `--normal-ratio`
- `--vip-weight`
- `--active-weight`
- `--normal-weight`

### 14.3 用户内部行为

- `--continue-prob`
- `--revive-policy`

### 14.4 到达时间与调度

- `--request-rate-per-user`
- `--arrival-distribution`
- `--max-parallel`
- `--selection-policy`
- `--seed`

### 14.5 终止与输出

- `--max-num-requests`
- `--log-file`
- `--output-details`
- `--tag`

## 15. 建议的实现结构

建议新增一个独立脚本，而不是复用旧的多轮 benchmark 逻辑。

建议文件组织如下：

- `benchmarks/schedule/tiering_ttft_bench.py`
- `benchmarks/schedule/tiering_ttft_dataset.py`
- `benchmarks/schedule/tiering_ttft_models.py`
- `benchmarks/schedule/tiering_ttft_scheduler.py`

如果第一版希望快速落地，也可以先单文件实现，但逻辑上仍建议拆分为以下几个核心对象。

### 15.1 `ConversationState`

职责：

- 保存真实对话 turn 序列
- 维护当前推进到的用户 turn
- 生成当前请求 payload
- 判断会话是否耗尽

### 15.2 `UserState`

职责：

- 持有多条会话
- 持有静态档位和权重
- 记录上一次活跃会话
- 维护 actor 状态
- 在请求完成后规划下一次请求

### 15.3 `QueuedRequest`

职责：

- 表示一个已经构造完成、等待进入调度的具体请求

建议字段：

- `request_id`
- `user_id`
- `conversation_id`
- `user_weight`
- `payload`
- `prompt_len`
- `enqueue_ts`
- `selection_reason`

### 15.4 `WeightedReadyQueue`

职责：

- 保存所有 ready 请求
- 支持按用户权重随机弹出请求

第一版可直接使用 Python list 加 `random.choices` 实现。

### 15.5 `UserActor`

职责：

- 在请求完成后进行泊松等待
- 选择下一条会话
- 构造下一次请求
- 将请求放入全局队列
- 保证同一用户至多一个 outstanding 请求

### 15.6 `BenchmarkScheduler`

职责：

- 在 `in_flight < max_parallel` 时持续调度
- 从队列中按权重选请求并发给服务端
- 记录 TTFT 和其他统计
- 请求完成后通知对应用户 actor
- 在达到终止条件后安全退出

## 16. 核心伪代码

### 16.1 用户 actor

```python
async def user_actor(user, scheduler):
    while not scheduler.should_stop() and not user.all_conversations_exhausted():
        await user.wait_for_previous_request_done()

        if scheduler.should_stop():
            break

        delay = sample_exponential(request_rate_per_user)
        user.state = "sleeping"
        await asyncio.sleep(delay)

        if scheduler.should_stop():
            break

        conv, reason = user.choose_next_conversation(continue_prob)
        if conv is None:
            user.state = "finished"
            break

        req = conv.build_queued_request(
            user_id=user.user_id,
            user_weight=user.weight,
            selection_reason=reason,
        )

        await scheduler.enqueue(req)
        user.state = "queued"
```

### 16.2 全局调度循环

```python
async def dispatch_loop(self):
    while not self.should_exit():
        while self.in_flight < self.max_parallel and not self.ready_queue.empty():
            req = self.ready_queue.pop_weighted()
            self.in_flight += 1
            self.mark_user_in_flight(req.user_id)
            asyncio.create_task(self.execute_request(req))

        await asyncio.sleep(0.001)
```

### 16.3 请求执行

```python
async def execute_request(self, req):
    start_ts = time.perf_counter()
    resp = await send_request(req.payload)
    finish_ts = time.perf_counter()

    self.record_metrics(
        req=req,
        ttft=resp.ttft,
        prompt_len=resp.prompt_len,
        cached_tokens=getattr(resp, "cached_tokens", None),
        queue_wait=start_ts - req.enqueue_ts,
        finish_ts=finish_ts,
    )

    self.advance_conversation(req.user_id, req.conversation_id)
    self.in_flight -= 1
    self.notify_user_done(req.user_id)
```

## 17. 实施阶段建议

### 第一阶段：最小可用版本

先实现以下能力：

- 真实多轮对话解析
- 一个用户对应多条真实对话
- continue / revive 两阶段会话选择
- 请求完成后的泊松休眠
- 全局 weighted ready queue
- `max_new_tokens = 1` 的 TTFT-only 测量
- 基于 `max_num_requests` 的终止条件

第一阶段暂不实现：

- 分档位不同的 think time
- 更复杂的 revive 权重策略
- 可视化与复杂统计图

### 第二阶段：增强观测能力

增加以下统计：

- continue / revive 的 TTFT 对比
- 各用户档位的请求贡献占比
- 各用户档位的缓存命中情况
- 队列等待时间分布

### 第三阶段：多层调度压力模式增强

增加以下能力：

- 热点阶段切换
- 特定档位更短的思考时间
- 更偏向冷会话唤醒的模式
- 按对话长度分层分配给用户

## 18. 验证方案

在正式用于多层调度实验前，应先验证 benchmark 自身是否满足预期。

### 18.1 用户档位偏斜验证

检查：

- VIP 用户是否确实贡献了更多已调度请求
- 普通用户是否没有被完全饿死

### 18.2 continue / revive 行为验证

检查：

- 实际 continue 比例是否接近设定值
- 被唤醒的历史会话是否持续有机会重新进入系统

### 18.3 前缀增长验证

检查：

- 同一会话推进时，prompt 长度是否逐步增长
- assistant 回复是否显著构成后续大前缀的主要部分

### 18.4 TTFT-only 验证

检查：

- 每次请求输出是否被严格压缩到最小
- benchmark 结果是否主要反映 TTFT，而非 decode 时间

### 18.5 多层调度敏感性验证

至少对比以下场景：

- 无缓存
- 启用缓存但无多层调度
- 启用缓存且启用多层调度

理想情况下，应能观察到：

- 热用户或热点会话的 TTFT 更低
- 资源受限时，尾延迟更稳定
- 被唤醒冷会话在下层恢复后，TTFT 有改善空间

## 19. 推荐默认参数

推荐第一版默认参数如下：

- `num_users = 128`
- `conversations_per_user = 4`
- `max_parallel = 32`
- `continue_prob = 0.8`
- `arrival_distribution = poisson`
- `request_rate_per_user = 0.1 ~ 0.5`
- `vip_ratio = 0.1`
- `active_ratio = 0.2`
- `normal_ratio = 0.7`
- `vip_weight = 8`
- `active_weight = 3`
- `normal_weight = 1`
- `max_new_tokens = 1`
- `max_num_requests = 1000`

这些参数应在第一次验证后，根据实际队列长度、prompt 增长速度和请求完成速率进行调整。

## 20. 最终建议

建议将这套 benchmark 作为 `benchmarks/schedule` 目录下独立的新工作负载来实现。

它的定位应当非常明确：

- 面向多层缓存调度
- 面向 TTFT
- 面向真实多用户、多会话、异步请求到达
- 面向前缀复用和冷会话回流

只要这几个目标保持清晰，这个 benchmark 就能成为后续评估多层调度策略是否有效的基础工具。
