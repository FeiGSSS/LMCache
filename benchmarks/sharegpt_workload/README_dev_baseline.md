# ShareGPT Baseline Test Plan (dev, no optimization)

## 目标
在做多层级调度优化前，先固定一套 `dev baseline`，作为后续所有策略对比的统一参照。

参考来源：
1. `origin/kv-quant` 分支的 quant benchmark 组织方式（"baseline 配置 + 固定工作负载 + 多轮统计 + 可复现 seed"）
2. 当前目录的 `sharegpt_multiturn_ttft_bench.py`（Phase-1: 1-token TTFT 专项）

## Baseline 定义
1. 代码基线：`upstream/dev` 官方实现（不含本地优化 patch）
2. 服务配置：使用 [lmcache_dev_baseline.yaml](/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/lmcache_dev_baseline.yaml)
3. 负载配置：使用 [scenarios_multiturn_ttft.yaml](/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml)
4. 输出约束：仅测 TTFT（`max_output_tokens=1`）

## 固定环境（建议）
1. 模型：`/home/fei/research/models/Qwen3-8B`
2. 数据：`/home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json`
3. Python：项目目录 `.venv`（uv 创建）
4. 服务端点：`http://127.0.0.1:8000/v1`

## 运行步骤
### 1) 启动 baseline 服务
在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=$(pwd)/benchmarks/sharegpt_workload/lmcache_dev_baseline.yaml \
vllm serve /home/fei/research/models/Qwen3-8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 2) 运行单场景（人工检查）

```bash
.venv/bin/python benchmarks/sharegpt_workload/sharegpt_multiturn_ttft_bench.py \
  --sharegpt-path /home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model /home/fei/research/models/Qwen3-8B \
  --scenario-file benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml \
  --scenario-name target_10_inflight_600s \
  --seed 41 \
  --progress-interval-sec 5 \
  --progress-summary-json outputs/sharegpt_baseline_target_seed41_progress_live.json \
  --output-csv outputs/sharegpt_baseline_target_seed41.csv \
  --summary-json outputs/sharegpt_baseline_target_seed41.json
```

### 3) 跑完整 baseline 矩阵（推荐）
使用 [run_dev_baseline_sharegpt.sh](/Users/philip/research/KVCache/LMCache/benchmarks/sharegpt_workload/run_dev_baseline_sharegpt.sh)。

```bash
bash benchmarks/sharegpt_workload/run_dev_baseline_sharegpt.sh
```

实时进度查看：

```bash
tail -f outputs/sharegpt_dev_baseline_*/target_10_inflight_600s_seed41_progress_live.json
```

## 场景矩阵（Phase-1）
1. 功能验证：`functional_serial`
2. 目标场景：`target_10_inflight_600s`
3. 到达率扫描：`scan_arrival_low/mid/high`
4. 并发闸门扫描：`scan_inflight_3/5/10/20`
5. 重复性：每场景 `seed=41,42,43`

## 核心指标与对比口径
1. 主指标：`ttft_server_p50/p90/p95/p99`
2. 并发闸门解释项：`ttft_effective_*`, `client_queue_wait_p50/p95`
3. 轮次分桶：`turn_1`, `turn_2_3`, `turn_4_plus`
4. 稳定性：同场景多 seed 的中位数与波动范围

## 通过标准（baseline 质量门槛）
1. `max_inflight_observed <= max_inflight_requests`
2. `status=success` 请求中 `ttft_server_sec` 与 `ttft_effective_sec` 非空
3. `turn_index` 对每个 `session_id` 单调递增且不跳轮
4. 中高负载下 `client_queue_wait_sec` 可观测上升

## 上游 dev 兼容跟踪（每轮实验前执行）

```bash
git fetch upstream dev
git rev-parse --short upstream/dev
git rev-list --left-right --count upstream/dev...HEAD
```

记录项：
1. `upstream/dev` commit
2. 当前分支 commit
3. 相对 `upstream/dev` 的 ahead/behind

## 备注
1. 该 baseline 只代表 Phase-1 的 prefill/缓存复用趋势，不代表完整 decode 场景。
2. Phase-2 再引入真实 decode（TPOT/E2E/吞吐）用于策略排序校准。
