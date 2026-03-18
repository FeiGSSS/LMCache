# Schedule Benchmark

本目录用于运行面向多层调度的 TTFT benchmark。

## 文件

- `benchmarks/schedule/src/convert_sharegpt.py`：转换脚本实现
- `benchmarks/schedule/dataset/`：转换后的 benchmark 数据集目录
- `benchmarks/schedule/lmcache_baseline.yaml`：LMCache 配置（当前 `max_local_cpu_size: 80`）
- `benchmarks/schedule/tiering_ttft_bench.py`：面向多层调度的 TTFT benchmark
- `benchmarks/schedule/tiering_ttft_benchmark_design.md`：TTFT benchmark 设计文档

## Step 0: 前置检查

在仓库根目录执行：

```bash
source .venv/bin/activate
```

## Step 1: 准备输入数据

默认命令就是我们推荐设置：`200` 条、按对话长度降序。

```bash
python benchmarks/schedule/src/convert_sharegpt.py \
  --output benchmarks/schedule/dataset/sharegpt_conv_top200_by_turns.json
```

可选：

```bash
python benchmarks/schedule/src/convert_sharegpt.py --count 200 --selection random
python benchmarks/schedule/src/convert_sharegpt.py --count 200 --selection length_asc
```

说明：

- 新的 TTFT benchmark 直接复用这里生成的数据格式
- 该格式为 `[{"id": ..., "messages": [...]}]`
- TTFT benchmark 会在运行时进一步过滤，只保留 `user turn >= 5` 的对话
- 如果希望更适合 TTFT benchmark，建议优先保留长对话，并适当增大 `--count`

## Step 2: 设置公共变量

```bash
export PORT=8000
export MODEL_PATH=/home/fei/research/models/Qwen3-8B
export INPUT_FILE=benchmarks/schedule/dataset/sharegpt_conv_top200_by_turns.json
mkdir -p benchmarks/schedule/dataset
mkdir -p benchmarks/schedule/results
```

## Step 3: 启动服务

纯 vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --enable-prompt-tokens-details
```

vLLM + LMCache:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
LMCACHE_CONFIG_FILE=benchmarks/schedule/lmcache_baseline.yaml \
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

服务就绪后，确认：

```bash
curl "http://localhost:$PORT/v1/models"
```

## Step 4: Flush Prefix Cache

建议每次正式跑 benchmark 之前，先清空 vLLM 的 prefix cache，保证不同实验从一致的冷启动状态开始。

因为 `/reset_prefix_cache` 是开发接口，启动服务时需要带上 `VLLM_SERVER_DEV_MODE=1`。

清空命令：

```bash
curl -X POST "http://localhost:$PORT/reset_prefix_cache"
```

如果需要同时清理 connector 相关缓存，可以使用：

```bash
curl -X POST "http://localhost:$PORT/reset_prefix_cache?reset_connector=true"
```

建议流程：

1. 启动服务
2. 确认 `v1/models` 接口正常
3. 执行一次 `/reset_prefix_cache`
4. 再启动 benchmark

如果希望 benchmark 报表中的 `平均缓存Tokens` 有值，启动服务时需要显式加上：

```bash
--enable-prompt-tokens-details
```

否则 vLLM 的 OpenAI 返回中不会包含 `usage.prompt_tokens_details.cached_tokens`。

## Step 5: Tiering TTFT Benchmark

这一部分用于测试多层缓存调度对 TTFT 的影响。

这个 benchmark 的核心建模是：

- 一个 `user` 表示一个真实用户，而不是一个简单发送器
- 每个用户持有多条私有真实历史对话
- 每次请求都使用真实 `Q/A` 历史推进，停在当前 user turn
- 每次请求只生成 `1 token`，尽量让 TTFT 成为主要观测指标
- 用户在请求完成后，按泊松分布等待一段时间，再决定继续当前会话或唤醒旧会话
- 请求进入全局队列后，调度器按所属用户权重做加权随机调度

### 运行入口

```bash
python3 benchmarks/schedule/tiering_ttft_bench.py \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --url "http://localhost:$PORT" \
  --input-file "$INPUT_FILE" \
  --num-users 128 \
  --conversations-per-user 4 \
  --min-user-turns 5 \
  --max-parallel 32 \
  --request-rate-per-user 0.2 \
  --continue-prob 0.8 \
  --seed 42 \
  --max-num-requests 1000 \
  --output-file benchmarks/schedule/results/tiering_ttft.json
```

### 关键参数

- `--input-file`：继续使用 `src/convert_sharegpt.py` 生成的数据文件；默认即指向 `benchmarks/schedule/dataset/sharegpt_conv_top200_by_turns.json`
- `--num-users`：用户数量，每个用户拥有自己的私有对话集
- `--conversations-per-user`：每个用户分配多少条真实历史对话
- `--min-user-turns`：过滤门槛，仅保留至少这么多 user turn 的对话，默认 `5`
- `--max-parallel`：系统同时在飞请求上限
- `--request-rate-per-user`：每个用户在请求完成后下一次入队的泊松到达率
- `--continue-prob`：用户继续当前会话的概率
- `--seed`：控制用户分档、对话分配、continue/revive、泊松等待和调度抽样的随机性
- `--max-num-requests`：总请求数上限
- `--output-file`：输出结果 JSON

### 默认行为

- 请求接口仍然是 vLLM 的 OpenAI 兼容接口 `/v1/chat/completions`
- 服务端启动方式和上面的实验保持一致，无需修改后端
- 默认 `max_tokens = 1`
- 默认用户分档与权重：
  - `vip = 10%`, `weight = 8`
  - `active = 20%`, `weight = 3`
  - `normal = 70%`, `weight = 1`

### 输出指标

脚本会在终端打印：

- `total_requests`
- `succeeded_requests`
- `failed_requests`
- `runtime_sec`
- `requests_per_sec`
- `mean_ttft_ms`
- `p50_ttft_ms`
- `p90_ttft_ms`
- `p99_ttft_ms`
- `mean_prompt_tokens`
- `mean_cached_tokens`
- `skipped_overlong_conversations`

同时还会输出分组统计：

- 按用户档位分组
- 按 `continue / revive` 分组
- 按 prompt 长度 bucket 分组

### 结果文件

如果指定 `--output-file`，会输出 JSON，包含：

- `summary`
- `by_tier`
- `by_reason`
- `prompt_buckets`
- `details`

其中 `details` 包含逐请求明细，便于后续离线分析。

如果某条会话继续推进后会超过模型上下文上限，benchmark 会在客户端侧直接跳过该会话，并在结果中累计到 `超长跳过会话数`。

### 使用建议

- 推荐继续使用按长度降序选出的长对话数据集
- 如果过滤后可用对话不足，请增大 `src/convert_sharegpt.py` 的 `--count`
- 三组服务对比时，应保持相同 `INPUT_FILE`、相同 `seed` 和相同 benchmark 参数
- 如果要观察多层调度效果，建议在较小 GPU 内存和启用 LMCache 的配置下运行

### 检查 cached_tokens 是否返回

可以用下面这段最小请求检查服务是否真的返回了 `cached_tokens`：

```bash
python3 - <<'PY'
import json
import urllib.request

url = "http://localhost:8001/v1/chat/completions"
payload = {
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "Repeat exactly: alpha beta gamma delta epsilon."}],
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
print(urllib.request.urlopen(request, timeout=30).read().decode())
PY
```

如果配置正确，返回中的 `usage` 应包含：

```json
"prompt_tokens_details": {
  "cached_tokens": ...
}
```
