# ShareGPT Multi-turn Concurrent TTFT Benchmark (Phase-1)

## 目标
本脚本用于 `TTFT` 专项压测，服务于 KV Cache 复用与后续多层级调度（CPU/磁盘/远端）策略评估。

Phase-1 设计约束：
1. 每次请求仅生成 `1 token`（强制）。
2. 会话历史中的 assistant 内容全部回放 ShareGPT 原文，不使用被测模型历史输出。
3. 并发定义为客户端在途异步请求数（`max_inflight_requests`），并分离：
   - `ttft_server_sec = first_token_time - dispatch_time`
   - `ttft_effective_sec = first_token_time - ready_time`
   - `client_queue_wait_sec = dispatch_time - submit_time(等待 semaphore 前后)`

## 新增文件
1. `/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/sharegpt_multiturn_ttft_bench.py`
2. `/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml`
3. `/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/README_multiturn_ttft.md`

## Baseline 方案
1. baseline 文档：`/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/README_dev_baseline.md`
2. baseline 配置：`/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/lmcache_dev_baseline.yaml`
3. baseline 批跑脚本：`/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/run_dev_baseline_sharegpt.sh`

## 数据与输入格式
- ShareGPT 文件为 JSON 数组，单条数据包含 `id` 和 `conversations`。
- 脚本从 `conversations` 中抽取交替 user/assistant 对；仅保留轮数 >= 2 的会话。

## 调度逻辑（已实现）
1. 新会话到达：Poisson，到达间隔 `Exp(new_user_rate)`。
2. 下一轮发起：上一轮完成后按
   - `mean = think_base_sec + assistant_ref_tokens / read_tok_per_sec`
   - 使用 Lognormal(`think_sigma`) 采样抖动。
3. 事件优先级：同时间戳时 `NEXT_TURN` 优先于 `NEW_SESSION`。
4. 运行终止：`duration_sec` 后不再接受新事件，排空在途请求再结束。

## CLI
必填参数：
1. `--sharegpt-path`
2. `--base-url`
3. `--model`

关键参数：
1. `--num-users`
2. `--max-inflight-requests`
3. `--new-user-rate`
4. `--duration-sec`
5. `--seed`

思考时间参数：
1. `--think-base-sec`
2. `--read-tok-per-sec`
3. `--think-sigma`

上下文与请求参数：
1. `--max-context-tokens`
2. `--max-output-tokens`（Phase-1 必须为 `1`）
3. `--request-timeout-sec`

输出参数：
1. `--output-csv`
2. `--summary-json`
3. `--scenario-file`
4. `--scenario-name`
5. `--progress-interval-sec`（实时进度打印间隔，秒）
6. `--progress-summary-json`（实时快照 JSON，可被 `tail` 监控）

参数优先级：
1. 同时提供 `--scenario-file` 与 CLI 参数时，CLI 显式传入值优先。
2. 场景文件仅用于填充未在 CLI 指定的参数。

## 示例
依赖（建议在 A800 的 uv 环境）：
```bash
uv pip install openai pyyaml
```

### 1) 功能测试（串行）
```bash
.venv/bin/python benchmarks/sharegpt_workload/sharegpt_multiturn_ttft_bench.py \
  --sharegpt-path /home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model /home/fei/research/models/Qwen3-8B \
  --scenario-file benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml \
  --scenario-name functional_serial \
  --output-csv outputs/functional_serial_requests.csv \
  --summary-json outputs/functional_serial_summary.json
```

### 2) 目标场景（10 inflight, 600s）
```bash
.venv/bin/python benchmarks/sharegpt_workload/sharegpt_multiturn_ttft_bench.py \
  --sharegpt-path /home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model /home/fei/research/models/Qwen3-8B \
  --scenario-file benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml \
  --scenario-name target_10_inflight_600s \
  --progress-interval-sec 5 \
  --progress-summary-json outputs/target_progress_live.json \
  --output-csv outputs/target_requests.csv \
  --summary-json outputs/target_summary.json
```

实时监控示例：

```bash
tail -f outputs/target_progress_live.json
```

### 3) 重复性测试（seed 扫描）
```bash
for s in 41 42 43; do
  .venv/bin/python benchmarks/sharegpt_workload/sharegpt_multiturn_ttft_bench.py \
    --sharegpt-path /home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --base-url http://127.0.0.1:8000/v1 \
    --model /home/fei/research/models/Qwen3-8B \
    --scenario-file benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml \
    --scenario-name target_10_inflight_600s \
    --seed ${s} \
    --output-csv outputs/target_seed${s}_requests.csv \
    --summary-json outputs/target_seed${s}_summary.json
done
```

## 输出
### 明细 CSV
每请求一行，字段与 `RequestRecord` 一致：
- `run_id, scenario, seed, session_id, conversation_id, turn_index`
- `ready_time, dispatch_time, first_token_time`
- `ttft_server_sec, ttft_effective_sec, client_queue_wait_sec`
- `prompt_tokens_est, status, error`

### 汇总 JSON
包含：
- `total_requests, success_rate, throughput_rps`
- `ttft_server_p50/p90/p95/p99`
- `ttft_effective_p50/p90/p95/p99`
- `client_queue_wait_p50/p95`
- `by_turn_index`（`turn_1`, `turn_2_3`, `turn_4_plus`）

## 验收检查建议
1. `max_inflight_observed <= max_inflight_requests`
2. 同一 `session_id` 的 `turn_index` 单调递增且不跳轮
3. `status=success` 的请求中 `ttft_server_sec` 与 `ttft_effective_sec` 非空
4. 中高负载下 `client_queue_wait_sec` 上升，并推动 `ttft_effective_sec` 上升

## Phase-2 TODO（未实现）
1. 真实 decode 轨道：`max_output_tokens` 按 ShareGPT `output_len` 分布采样（带上限）
2. 新增指标：`TPOT`, `E2E latency`, `output throughput`
3. 校准判据：比较 Phase-1 与 Phase-2 的策略排序一致性

## 注意
仅生成 `1 token` 会弱化 decode 阶段影响，结论主要用于 prefill/缓存复用与调度策略初筛，不代表完整端到端服务表现。
