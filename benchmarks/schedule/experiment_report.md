# Schedule TTFT 实验记录

## 实验目标

记录 `benchmarks/schedule/tiering_ttft_bench.py` 在不同 vLLM / LMCache 配置下的 TTFT 基线结果，作为后续多层调度实验的对照。

## Benchmark 脚本

本次使用的 benchmark 命令如下：

```bash
python3 benchmarks/schedule/tiering_ttft_bench.py \
  --model /data/llm-models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --url http://localhost:8001 \
  --input-file benchmarks/schedule/dataset/sharegpt_conv_all_min5.json \
  --num-users 128 \
  --conversations-per-user 10 \
  --min-user-turns 5 \
  --max-parallel 32 \
  --request-rate-per-user 1 \
  --continue-prob 0.8 \
  --seed 42 \
  --max-num-requests 4000 \
  --output-file benchmarks/schedule/results/tiering_ttft_vllm_full.json
```

## 实验 1：纯 vLLM，低 GPU KV 容量基线

### 服务启动命令

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
vllm serve /data/llm-models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.4 \
  --enable-prompt-tokens-details
```

### Prefix cache reset

```bash
curl -X POST "http://localhost:8001/reset_prefix_cache"
```

### 结果

```text
多层调度 TTFT Benchmark 汇总
总请求数                 4000
成功请求数               4000
失败请求数               0
总运行时间(秒)           333.207
吞吐(req/s)              12.005
平均TTFT(ms)             2638.043
P50 TTFT(ms)             2867.323
P90 TTFT(ms)             4185.110
P99 TTFT(ms)             5478.113
平均Prompt Tokens        1387.885
平均缓存Tokens           228.488
超长跳过会话数           1

按用户档位统计
active             1090       2707.657       4240.061
normal             2146       2555.019       4165.941
vip                 764       2771.932       4193.646

按会话选择方式统计
continue           2998       2708.166       4192.676
revive             1002       2428.233       4161.123

按Prompt长度分桶统计
1k-4k              1587       2921.079       4208.360
4k-8k               232       3431.964       4824.342
<1k                2161       2331.771       4133.123
>=8k                 20       4062.356       4981.453
```

### 初步观察

- 在 `--gpu-memory-utilization 0.4` 下，纯 vLLM 的 TTFT 明显上升。
- `平均缓存Tokens = 228.488`，相对高缓存容量场景显著下降，说明本地可保留前缀明显不足。
- `P90 / P99` 非常高，说明长尾延迟已经很严重。
- 该结果可作为后续 `vLLM + LMCache`、以及更进一步多层调度实验的低显存基线。

## 实验 2：vLLM + LMCache + Tiering（命令行直配）

### 说明

如果要测试多层调度，不能只开 CPU backend，还需要显式打开 disk backend。

这里不使用 yaml，而是直接在命令行环境变量中配置 LMCache：

- `LMCACHE_CHUNK_SIZE=256`
- `LMCACHE_LOCAL_CPU=True`
- `LMCACHE_MAX_LOCAL_CPU_SIZE=80`
- `LMCACHE_LOCAL_DISK=file:///tmp/lmcache_schedule_disk/`
- `LMCACHE_MAX_LOCAL_DISK_SIZE=200`
- `LMCACHE_ENABLE_TIERING=True`

> `local_disk` 需要是 `file://` 形式的目录路径。

### 服务启动命令

```bash
mkdir -p /tmp/lmcache_schedule_disk

CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=80 \
LMCACHE_LOCAL_DISK=file:///tmp/lmcache_schedule_disk/ \
LMCACHE_MAX_LOCAL_DISK_SIZE=200 \
LMCACHE_ENABLE_TIERING=True \
vllm serve /data/llm-models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.4 \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Prefix cache reset

```bash
curl -X POST "http://localhost:8001/reset_prefix_cache"
```

### Benchmark 命令

```bash
python3 benchmarks/schedule/tiering_ttft_bench.py \
  --model /data/llm-models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --url http://localhost:8001 \
  --input-file benchmarks/schedule/dataset/sharegpt_conv_all_min5.json \
  --num-users 128 \
  --conversations-per-user 10 \
  --min-user-turns 5 \
  --max-parallel 32 \
  --request-rate-per-user 1 \
  --continue-prob 0.8 \
  --seed 42 \
  --max-num-requests 4000 \
  --output-file benchmarks/schedule/results/tiering_ttft_lmcache_tiering_full.json
```
