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

## 实验 2：vLLM + LMCache（CPU 30G，官方仓库验证）

### 说明

该实验使用官方 LMCache 仓库中的 connector 实现进行验证，只开启 CPU backend，不开启 disk tiering，CPU 本地缓存容量为 `30G`。

这组结果的意义是先确认：

- 当前 `benchmarks/schedule/tiering_ttft_bench.py` 可以在官方 `vLLM + LMCache` 组合下稳定跑通。
- 在相同 `--gpu-memory-utilization 0.4` 条件下，LMCache 是否能显著改善 TTFT 与吞吐。

### 服务启动命令

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=30 \
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
  --output-file benchmarks/schedule/results/tiering_ttft_lmcache_cpu_30g_full.json
```

### 结果

```text
多层调度 TTFT Benchmark 汇总
总请求数                 4000
成功请求数               4000
失败请求数               0
总运行时间(秒)           182.426
吞吐(req/s)              21.927
平均TTFT(ms)             1443.890
P50 TTFT(ms)             1427.396
P90 TTFT(ms)             2108.269
P99 TTFT(ms)             2921.018
平均Prompt Tokens        1360.384
平均缓存Tokens           839.016
超长跳过会话数           1

按用户档位统计
active             1066       1462.088       2110.251
normal             2256       1426.469       2112.953
vip                 678       1473.248       2095.123

按会话选择方式统计
continue           3000       1478.160       2110.331
revive             1000       1341.082       2051.828

按Prompt长度分桶统计
1k-4k              1579       1568.871       2197.602
4k-8k               216       1798.541       2634.168
<1k                2188       1314.001       2001.408
>=8k                 17       2046.795       2891.837
```

### 初步观察

- 在官方 `vLLM + LMCache` 组合下，benchmark 可以稳定完成 `4000/4000` 请求，说明此前出现的 `EngineDeadError` 不能简单归因于 benchmark 逻辑本身。
- 相比实验 1 的纯 vLLM 低显存基线，吞吐从 `12.005 req/s` 提升到 `21.927 req/s`，仍接近翻倍。
- 平均 TTFT 从 `2638.043 ms` 降到 `1443.890 ms`，显著下降，说明 LMCache 在该 workload 下有效缓解了前缀缓存容量不足带来的性能退化。
- `平均缓存Tokens = 839.016`，显著高于纯 vLLM 基线的 `228.488`，说明更多历史前缀被有效复用。
- `continue` 的平均 TTFT 高于 `revive`，与其通常对应更长历史前缀、prompt 更长的现象一致。
- prompt 长度越长，TTFT 越高，`<1k` 到 `>=8k` 呈现出符合预期的单调恶化趋势。

## 补充结果：vLLM + LMCache（CPU 40G，官方仓库验证）

### 服务启动命令

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=40 \
vllm serve /data/llm-models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.4 \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 结果

```text
多层调度 TTFT Benchmark 汇总
总请求数                 4000
成功请求数               4000
失败请求数               0
总运行时间(秒)           167.986
吞吐(req/s)              23.811
平均TTFT(ms)             1332.318
P50 TTFT(ms)             1314.014
P90 TTFT(ms)             1815.273
P99 TTFT(ms)             2784.276
平均Prompt Tokens        1399.956
平均缓存Tokens           937.680
超长跳过会话数           1

按用户档位统计
active             1029       1343.275       1806.414
normal             2302       1320.877       1813.111
vip                 669       1354.831       1830.205

按会话选择方式统计
continue           3014       1362.964       1823.853
revive              986       1238.637       1794.635

按Prompt长度分桶统计
1k-4k              1570       1432.253       1861.569
4k-8k               229       1595.453       2076.990
<1k                2172       1225.820       1732.282
>=8k                 29       1820.422       2754.549
```

### 初步观察

- 相比 `CPU 30G`，`CPU 40G` 的吞吐更高、TTFT 更低，说明在该 workload 下增大 CPU 本地缓存容量仍能继续提升命中收益。
- `平均缓存Tokens` 从 `839.016` 提升到 `937.680`，与性能改善方向一致。

## 后续：Tiering 实验说明

如果要继续测试多层调度，不能只开 CPU backend，还需要显式打开 disk backend，并启用 tiering。

这里不使用 yaml，而是直接在命令行环境变量中配置 LMCache：

- `LMCACHE_CHUNK_SIZE=256`
- `LMCACHE_LOCAL_CPU=True`
- `LMCACHE_MAX_LOCAL_CPU_SIZE=80`
- `LMCACHE_LOCAL_DISK=file:///tmp/lmcache_schedule_disk/`
- `LMCACHE_MAX_LOCAL_DISK_SIZE=200`
- `LMCACHE_ENABLE_TIERING=True`

> `local_disk` 需要是 `file://` 形式的目录路径。
