# KV Cache 量化精度测试

使用 GSM8K 数据集测试 KIVI 风格 KV 缓存量化的精度影响。

## 配置

- **量化**: 4-bit KIVI 风格 (group_size=128)
- **数据集**: GSM8K (1319 条测试样本)
- **模型**: Qwen3-8B

## 注意事项

⚠️ **关于时间**: 本测试中的时间没有太大意义。

- 量化的优化点是 **TTFT (Time To First Token)**，主要针对 **长 prompt** 场景
- GSM8K 数据集的平均输入长度约为 **900 tokens**，属于短 prompt
- 如果要测试量化的加速效果，应该使用长文档场景 (如 LongBench)

📝 **Phase 1 的双重作用**:

Phase 1 (Baseline) 阶段一石二鸟，同时完成两件事：

1. **计算全精度分数**: 由于缓存完全 miss，vLLM 使用全精度计算输出，然后 LMCache 将 KV 缓存量化和存储
2. **生成量化缓存**: 第一轮生成了量化后的 KV 缓存，为 Phase 2 做准备

💾 **关于缓存卸载**:

- Qwen3-8B 的 KV 缓存约 **130 GB**
- 第一轮生成的缓存量远超 GPU 显存限制
- 所有缓存会被完全卸载到 **CPU**，避免第二轮在 GPU 上命中全精度缓存

## 准备工作

1. 安装依赖:
```bash
pip install datasets tiktoken openai vllm pyarrow
```

2. 准备数据集目录结构:
```
/path/to/datasets/
└── gsm8k/
    └── main/
        ├── test-00000-of-00001.parquet
        └── train-00000-of-00001.parquet
```

3. 准备 prompt 文件 (用于 few-shot 评估):
```
./lib_prompts/
└── gsm8k_prompt_original.txt
```

## 运行测试

### 步骤 1: 启动带量化的 vLLM 服务器

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=$(pwd)/benchmarks/quantization/acc/quant_acc_test_config.yaml \
vllm serve /path/to/Qwen3-8B/ \
  --tensor-parallel-size 1 \
  --port 9001 \
  --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'
```

### 步骤 2: 运行精度评估

```bash
python benchmarks/quantization/acc/quantization_test.py \
  --dataset gsm8k \
  --model /path/to/Qwen3-8B/ \
  --port 9001
```

#### 自定义路径 (如不使用默认路径)

```bash
python benchmarks/quantization/acc/quantization_test.py \
  --dataset gsm8k \
  --model /path/to/Qwen3-8B/ \
  --port 9001 \
  --datasets-dir /path/to/datasets \
  --prompt-dir $(pwd)/benchmarks/quantization/acc/lib_prompts
```

## 预期输出

```
INFO: Loaded 1319 samples for evaluation.
INFO: Phase 1: Baseline Evaluation...
Evaluating: 100%|████████████████████████████████████████| 1319/1319 [03:36<00:00,  6.08samples/s]
INFO: Baseline accuracy: 0.9333
INFO: Phase 2: Quantized Evaluation...
Evaluating: 100%|████████████████████████████████████████| 1319/1319 [02:13<00:00,  9.87samples/s]
INFO: Quantized accuracy: 0.9287
```

## 结果解读

### Qwen3-8B 实验结果

| 指标 | 数值 |
|------|------|
| Baseline 精度 | 93.33% |
| Quantized 精度 | 92.87% |
| 精度下降 | -0.46% |

精度下降约 0.5%，说明 4-bit KIVI 量化对 Qwen3-8B 在 GSM8K 上影响很小。

### Mistral-7B-Instruct-v0.2 实验结果

| 指标 | 数值 |
|------|------|
| Baseline 精度 | 33.21% |
| Quantized 精度 | 32.83% |
| 精度下降 | -0.38% |

Mistral-7B-Instruct-v0.2 在 GSM8K 上的绝对精度较低 (约 33%)，可能是模型本身的数学推理能力较弱。4-bit 量化对该模型的精度影响同样很小 (-0.38%)。

## 工作流程

1. **Phase 1 (Baseline)**: 首次评估所有样本缓存完全 miss
2. **Phase 2 (Quantized)**: 再次评估，KV 缓存以量化格式存储在 CPU，从 CPU 加载后 GPU 解量化

## 自定义量化配置

修改 `quant_acc_test_config.yaml` 可测试不同量化参数:

```yaml
enable_kv_quantization: true
kv_quantization_bits: 4       # 2, 4, 或 8
kv_quantization_group_size: 128  # 32, 64, 128, 256
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | 模型路径或名称 |
| `--dataset` | gsm8k | 数据集名称 |
| `--port` | 8000 | vLLM 服务器端口 |
| `--base-url` | http://localhost:8000/v1 | 完整服务器 URL |
| `--datasets-dir` | /home/fei/research/datasets | 数据集目录 |
| `--prompt-dir` | ./lib_prompts | prompt 文件目录 |
| `--max-new-tokens` | 512 | 最大生成 token 数 |
| `--concurrency` | 30 | 并发 API 请求数 |
| `--zero-shot` | False | 使用 zero-shot 评估 |
