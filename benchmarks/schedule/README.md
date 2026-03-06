# Schedule Benchmark

本目录用于复现单卡 Qwen3-8B 的多轮对话对比实验，比较三组配置：

- 纯 vLLM（高显存）
- 纯 vLLM（低显存）
- vLLM + LMCache（低显存 + CPU DRAM）

## 文件

- `benchmarks/schedule/convert_sharegpt.py`：生成 benchmark 输入数据
- `benchmarks/schedule/lmcache_baseline.yaml`：LMCache 配置（当前 `max_local_cpu_size: 80`）

## Step 0: 前置检查

在仓库根目录执行：

```bash
source .venv/bin/activate
```

## Step 1: 准备输入数据

默认命令就是我们推荐设置：`200` 条、按对话长度降序。

```bash
python benchmarks/schedule/convert_sharegpt.py \
  --output benchmarks/schedule/sharegpt_conv_top200_by_turns.json
```

可选：

```bash
python benchmarks/schedule/convert_sharegpt.py --count 200 --selection random
python benchmarks/schedule/convert_sharegpt.py --count 200 --selection length_asc
```

## Step 2: 设置公共变量

```bash
export PORT=8000
export MODEL_PATH=/home/fei/research/models/Qwen3-8B
export VLLM_BENCH=/home/fei/research/llm/KVCache/vllm/benchmarks/multi_turn
export INPUT_FILE=benchmarks/schedule/sharegpt_conv_top200_by_turns.json
mkdir -p benchmarks/schedule/results
```

## Step 3: 运行三组实验

每组实验都按相同流程执行：

1. 启动服务
2. 等待服务就绪
3. 运行 benchmark
4. 停服务

统一请求命令（所有实验共用）：

```bash
python "$VLLM_BENCH/benchmark_serving_multi_turn.py" \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --url "http://localhost:$PORT" \
  --input-file "$INPUT_FILE" \
  --num-clients 4 \
  --max-active-conversations 200 \
  --conversation-sampling round_robin \
  --max-turns 10 \
  --seed 42 \
  --no-early-stop \
  --output-file "$OUTPUT_FILE"
```

参数含义（这条请求在模拟什么）：

- 心智模型：`client` 不是单个用户，而是一个并发发送器。每个 `client` 内部维护一组用户会话列表，并按给定策略选择“下一位要服务的用户”发下一轮请求。多个 `client` 并行工作。
- `--input-file "$INPUT_FILE"`：多轮对话样本池。这里用 Top-200 长对话，模拟“上下文较长、KV 压力更高”的线上会话。
- `--num-clients 4`：并发客户端数量。表示有 4 个发送器并行推进各自维护的用户群。
- `--max-active-conversations 200`：全局活跃用户上限，统计范围是所有客户端合计，不是单个客户端。
- `--conversation-sampling round_robin`：单个客户端内部的选人策略。这里是轮询，从该客户端维护的用户列表中依次选择下一位用户发请求。
- 每次某个用户返回响应后，客户端会先把这次 assistant 回复写回该用户会话上下文（用于下一轮请求）。
- 然后按条件二选一：
- 继续：如果该用户还没达到结束条件，会把该用户重新 append 回活跃队列，等待下一次被调度。
- 结束：如果已达到结束条件（`turns_count >= min(max_turns, len(messages))`），该用户会话从活跃集合移除，不再被调度。
- `--max-turns 10`：每个会话的上限（源码按 message 计数并与 `len(messages)` 共同裁剪），用于限制单个用户会话推进深度。
- `--no-early-stop`：即使达到某些局部停止条件也继续跑完整体负载，保证三组实验的可比性和稳定性。
- `--seed 42`：固定采样随机性，保证多次实验输入顺序可复现。
- `--output-file "$OUTPUT_FILE"`：每组实验写入不同结果文件，避免覆盖，便于横向对比。

### 3.1 Baseline A: vLLM 高显存

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

新开一个终端执行请求：

```bash
OUTPUT_FILE=benchmarks/schedule/results/vllm_high_mem.json
python "$VLLM_BENCH/benchmark_serving_multi_turn.py" \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --url "http://localhost:$PORT" \
  --input-file "$INPUT_FILE" \
  --num-clients 4 \
  --max-active-conversations 200 \
  --conversation-sampling round_robin \
  --max-turns 10 \
  --seed 42 \
  --no-early-stop \
  --output-file "$OUTPUT_FILE"
```

### 3.2 Baseline B: vLLM 低显存

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35
```

请求命令：

```bash
OUTPUT_FILE=benchmarks/schedule/results/vllm_low_mem.json
python "$VLLM_BENCH/benchmark_serving_multi_turn.py" \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --url "http://localhost:$PORT" \
  --input-file "$INPUT_FILE" \
  --num-clients 4 \
  --max-active-conversations 200 \
  --conversation-sampling round_robin \
  --max-turns 10 \
  --seed 42 \
  --no-early-stop \
  --output-file "$OUTPUT_FILE"
```

### 3.3 Experiment C: vLLM + LMCache

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=benchmarks/schedule/lmcache_baseline.yaml \
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

请求命令：

```bash
OUTPUT_FILE=benchmarks/schedule/results/vllm_lmcache_low_mem.json
python "$VLLM_BENCH/benchmark_serving_multi_turn.py" \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --url "http://localhost:$PORT" \
  --input-file "$INPUT_FILE" \
  --num-clients 4 \
  --max-active-conversations 200 \
  --conversation-sampling round_robin \
  --max-turns 10 \
  --seed 42 \
  --no-early-stop \
  --output-file "$OUTPUT_FILE"
```

## Step 4: 对比结果

重点看：

- `ttft_ms`：`mean` / `p90` / `p99`
- `requests_per_sec`
- server 日志里的 `Prefix cache hit rate` / `External prefix cache hit rate`

## 常见坑与规避
- 必须单卡参数一致：`CUDA_VISIBLE_DEVICES=0` 且 `--tensor-parallel-size 1`。
- 服务未就绪就发请求会失败：先确认 `curl http://localhost:$PORT/v1/models` 返回正常。
- 同时起多个服务会端口冲突：每次实验结束后先 `Ctrl+C` 停掉当前服务。
- 不要删 `--no-early-stop`：多轮压测下更稳定，避免提前停止造成结果偏差。
- 三组实验使用同一个 `INPUT_FILE`：否则结果不可比。

## 可选参数

需要限制请求总数时，在 benchmark 命令后追加：

```bash
--max-num-requests 200
```
