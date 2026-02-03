# KV Cache 量化精度测试

使用 GSM8K 数据集测试 KIVI 风格 KV 缓存量化的精度影响。

## 概述

- **量化**: 4-bit KIVI 风格 (group_size=128)
- **数据集**: GSM8K (1319 条测试样本)
- **模型**: Qwen3-8B, Mistral-7B-Instruct-v0.2

## 实验结果

### Qwen3-8B

| 配置 | 精度 | 下降 |
|------|------|------|
| Baseline | 93.33% | - |
| 4-bit 量化 | 92.87% | -0.46% |

### Mistral-7B-Instruct-v0.2

| 配置 | 精度 | 下降 |
|------|------|------|
| Baseline | 33.21% | - |
| 4-bit 量化 | 32.83% | -0.38% |

**结论**: 4-bit 量化对 GSM8K 精度影响很小 (<0.5%)

## 准备工作

```bash
pip install datasets tiktoken openai vllm pyarrow
```

数据集目录结构:
```
/path/to/datasets/
└── gsm8k/
    └── main/
        └── test-00000-of-00001.parquet
```

## 运行测试

### 步骤 1: 启动 vLLM 服务器

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=$(pwd)/benchmarks/quantization/acc/quant_acc_test_config.yaml \
vllm serve /home/fei/research/models/Qwen3-8B/ \
  --tensor-parallel-size 1 \
  --port 9001 \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'
```

### 步骤 2: 运行精度评估

```bash
python benchmarks/quantization/acc/quantization_test.py \
  --dataset gsm8k \
  --model /home/fei/research/models/Qwen3-8B/ \
  --port 9001
```

#### 自定义路径

```bash
python benchmarks/quantization/acc/quantization_test.py \
  --dataset gsm8k \
  --model /home/fei/research/models/Qwen3-8B/ \
  --port 9001 \
  --datasets-dir /path/to/datasets \
  --prompt-dir $(pwd)/benchmarks/quantization/acc/lib_prompts
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | 模型路径 |
| `--dataset` | gsm8k | 数据集名称 |
| `--port` | 8000 | 服务器端口 |
| `--datasets-dir` | /home/fei/research/datasets | 数据集目录 |
| `--prompt-dir` | ./lib_prompts | prompt 文件目录 |
| `--max-new-tokens` | 512 | 最大生成 token 数 |
| `--concurrency` | 30 | 并发请求数 |
| `--zero-shot` | False | zero-shot 评估 |

## 注意事项

⚠️ **关于时间**: 本测试时间无意义，量化优化 TTFT 针对长 prompt，GSM8K 平均仅 900 tokens

📝 **Phase 1 双重作用**:
- 计算全精度分数 (缓存 miss)
- 生成量化缓存 (为 Phase 2 准备)

💾 **缓存卸载**: 130GB KV 缓存 > GPU 显存，完全卸载到 CPU

## 自定义量化配置

修改 `quant_acc_test_config.yaml`:

```yaml
enable_kv_quantization: true
kv_quantization_bits: 4       # 2, 4, 或 8
kv_quantization_group_size: 128
```
