# KV Cache 量化 TTFT 测试

测试 KIVI 风格 KV 缓存量化对 TTFT (Time To First Token) 的优化效果。

## 概述

- **量化**: 4-bit KIVI 风格 (group_size=128)
- **测试工具**: `benchmarks/long_doc_qa/long_doc_qa.py`
- **测试场景**: 长文档 (15 docs × 40K tokens = 600K tokens)

## 准备工作

```bash
pip install datasets tiktoken openai vllm
```

## 实验结果

### Qwen3-8B

| 配置 | Warmup TTFT | Query TTFT | 加速比 |
|------|-------------|------------|--------|
| Baseline (无量化) | 11.441s | 1.126s | 10.16× |
| 4-bit 量化 | 11.646s | 0.560s | 20.80× |

**量化额外加速 2.01×** (Query TTFT: 1.126s → 0.560s)

## 运行测试

需要分别测试 **有量化** 和 **无量化** 两种配置。

### 测试命令 (两种配置通用)

```bash
python benchmarks/long_doc_qa/long_doc_qa.py \
  --model /home/fei/research/models/Qwen3-8B/ \
  --num-documents 15 \
  --document-length 40000 \
  --output-len 100 \
  --repeat-count 1 \
  --repeat-mode tile \
  --max-inflight-requests 4 \
  --port 9001
```

### 测试 1: 无量化 (Baseline)

```bash
# 启动服务器
CUDA_VISIBLE_DEVICES=1 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=$(pwd)/benchmarks/quantization/ttft/ttft_baseline_config.yaml \
vllm serve /home/fei/research/models/Qwen3-8B/ \
  --tensor-parallel-size 1 \
  --port 9001 \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'

# 运行测试 (使用上述测试命令)
```

### 测试 2: 4-bit 量化

```bash
# 启动服务器
CUDA_VISIBLE_DEVICES=1 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=$(pwd)/benchmarks/quantization/ttft/ttft_quant_config.yaml \
vllm serve /home/fei/research/models/Qwen3-8B/ \
  --tensor-parallel-size 1 \
  --port 9001 \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'

# 运行测试 (使用上述测试命令)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-documents` | 15 | 采样文档数量 |
| `--document-length` | 40000 | 每文档 tokens 数 |
| `--output-len` | 100 | 生成 token 数 |
| `--repeat-count` | 1 | 每 prompt 重复次数 |
| `--repeat-mode` | tile | 重复模式 |
| `--max-inflight-requests` | 4 | 最大并发请求数 |
| `--port` | 9001 | vLLM 服务器端口 |

## 测试流程

1. **Warmup 阶段**: 首次处理长文档，缓存 miss，计算 TTFT
2. **Query 阶段**: 重复相同请求，缓存 hit，对比 TTFT 提升

## 注意事项

💾 **CPU 缓存大小**: 150GB，足够容纳 ~100GB+ 的 KV 缓存

📝 **测试流程**:
- Warmup 阶段: 缓存 miss，全精度计算，量化和存储到 CPU
- Query 阶段: 缓存 hit，从 CPU 加载量化数据，GPU 解量化

⚠️ **端口管理**: 确保两次测试使用不同端口

## 配置文件

| 文件 | 说明 |
|------|------|
| `ttft_baseline_config.yaml` | 无量化配置 |
| `ttft_quant_config.yaml` | 4-bit 量化配置 |

### 自定义量化位数

修改 `ttft_quant_config.yaml`:

```yaml
enable_kv_quantization: true
kv_quantization_bits: 2       # 2, 4, 或 8
kv_quantization_group_size: 128
```
