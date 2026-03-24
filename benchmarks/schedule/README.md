# Schedule Benchmark

本目录用于运行面向多层调度的 TTFT benchmark。

## 文件

- `benchmarks/schedule/src/convert_sharegpt.py`：转换脚本实现
- `benchmarks/schedule/dataset/`：转换后的 benchmark 数据集目录
- `benchmarks/schedule/serve.sh`：按场景启动 vLLM 服务的脚本
- `benchmarks/schedule/bench.sh`：按场景运行 TTFT benchmark 的脚本
- `benchmarks/schedule/tiering_ttft_bench.py`：面向多层调度的 TTFT benchmark
- `benchmarks/schedule/schedule_benchmark_modeling.md`：统一的 benchmark 建模与机制说明文档

## Step 1: 准备输入数据

默认命令就是推荐设置：`200` 条、按对话长度降序。

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

- 数据格式为 `[{"id": ..., "messages": [...]}]`
- TTFT benchmark 会进一步过滤，只保留 `user turn >= 5` 的对话
- 如果过滤后可用对话不足，请增大 `--count`

## 快速开始

脚本会自动激活当前项目的 `.venv`。

支持的场景：

- `cpu`：GPU `0`，port `8001`，仅启用 CPU backend
- `cpu_disk`：GPU `1`，port `8002`，启用 CPU + Disk，关闭 tiering
- `cpu_disk_tiering`：GPU `2`，port `8003`，启用 CPU + Disk + tiering

对应 disk 目录：

- `cpu_disk`：`/tmp/lmcache_schedule_disk_cpu_disk/`
- `cpu_disk_tiering`：`/tmp/lmcache_schedule_disk_cpu_disk_tiering/`

### CPU

启动服务：

```bash
./benchmarks/schedule/serve.sh cpu
```

运行 benchmark：

```bash
./benchmarks/schedule/bench.sh cpu
```

### CPU + Disk

启动服务：

```bash
./benchmarks/schedule/serve.sh cpu_disk
```

运行 benchmark：

```bash
./benchmarks/schedule/bench.sh cpu_disk
```

### CPU + Disk + Tiering

启动服务：

```bash
./benchmarks/schedule/serve.sh cpu_disk_tiering
```

运行 benchmark：

```bash
./benchmarks/schedule/bench.sh cpu_disk_tiering
./benchmarks/schedule/bench.sh cpu_disk_tiering --num-requests 2200
```

输出文件会按场景自动命名到 `benchmarks/schedule/results/`。

## Benchmark 行为

这个 benchmark 的核心建模是：

- 一个 `user` 表示一个真实用户，而不是一个简单发送器
- 每个用户持有多条私有真实历史对话
- 每次请求都使用真实 `Q/A` 历史推进，停在当前 user turn
- 每次请求只生成 `1 token`，尽量让 TTFT 成为主要观测指标
- 用户在请求完成后，按泊松分布等待一段时间，再决定继续当前会话或唤醒旧会话
- 请求进入全局队列后，调度器按所属用户权重做加权随机调度

详细建模说明见 `benchmarks/schedule/schedule_benchmark_modeling.md`。

## 固定 benchmark 参数

`bench.sh` 当前固定使用：

- `--model /data/llm-models/Qwen3-8B`
- `--served-model-name Qwen3-8B`
- `--input-file benchmarks/schedule/dataset/sharegpt_conv_all_min5.json`
- `--num-users 128`
- `--conversations-per-user 10`
- `--min-user-turns 5`
- `--max-parallel 32`
- `--request-rate-per-user 1`
- `--continue-prob 0.8`
- `--seed 42`
- `--max-num-requests <通过 --num-requests 指定，默认 7000>`

## 输出指标

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

如果指定 `--output-file`，结果 JSON 包含：

- `summary`
- `by_tier`
- `by_reason`
- `prompt_buckets`
- `details`
