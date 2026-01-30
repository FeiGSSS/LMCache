# Retrieval 与量化反量化流程开发总结

> 目标：梳理当前 Retrieval 路径中“量化分支”的端到端流程、关键数据结构、关键接口与约束，方便后续维护与优化。

## 1. 总览：Retrieval 主流程

入口为 [lmcache/v1/cache_engine.py](lmcache/v1/cache_engine.py) 的 `retrieve()`。

**流程摘要（量化分支关注）：**
1. `retrieve()` 解析 `tokens` 与 `mask`，调用内部流程 `_process_tokens_internal()`（或异步路径）获取命中的 `reordered_chunks`。
2. 若开启量化（`config.enable_kv_quantization=True`），`retrieve()` 会把 `kv_quantization_bits` 与 `kv_quantization_group_size` 透传给 GPU Connector。
3. `retrieve()` 调用 `gpu_connector.batched_to_gpu(...)`，逐块触发 `to_gpu()`。
4. 在 `to_gpu()` 中识别量化 MemoryObj，走“量化反量化路径”：
   - 从多张量 MemoryObj 取出 6 个量化张量；
   - 调用 `lmc_ops.multi_layer_kv_transfer(...)` 的 TensorList 分支；
   - C++/CUDA op 在 GPU 上完成反量化并直接写入 vLLM paged KV。
5. 完成后 `retrieve()` 更新统计、处理引用计数与可选清理。

**核心入口链路：**
- `retrieve()` → `gpu_connector.batched_to_gpu()` → `VLLMPagedMemGPUConnectorV2.to_gpu()` → `lmc_ops.multi_layer_kv_transfer()` → C++/CUDA 反量化写入。

## 2. 量化 MemoryObj 布局与格式

量化对象使用 6 张量布局（KIVI-style），按顺序为：
1. `k_encoded`（int32）
2. `k_scale`（fp16/bf16）
3. `k_mn`（fp16/bf16）
4. `v_encoded`（int32）
5. `v_scale`（fp16/bf16）
6. `v_mn`（fp16/bf16）

**布局含义：**
- K 量化沿 token 维度分组；V 量化沿 head_dim 维度分组。
- 量化数据存储在 CPU pinned 或 GPU 上，支持 UVA 访问。

## 3. 量化 Retrieval 关键逻辑

### 3.1 `retrieve()` 中的量化参数透传
位置：[lmcache/v1/cache_engine.py](lmcache/v1/cache_engine.py)

- 当 `config.enable_kv_quantization=True` 时：
  - 将 `kv_quantization_bits` 与 `kv_quantization_group_size` 注入 `kwargs`。
  - 这些参数用于 GPU 侧反量化 kernel。

### 3.2 GPU Connector 的量化分支
位置：[lmcache/v1/gpu_connector.py](lmcache/v1/gpu_connector.py)

**核心逻辑：**
- `VLLMPagedMemGPUConnectorV2.to_gpu()` 中，优先检测 `memory_obj.metadata.is_quantized`。
- 若为量化：
  - 读取 6 个张量并调用 `lmc_ops.multi_layer_kv_transfer`，传入 `quantized=True` 与 bits/group_size。
  - 该调用在 GPU 上直接完成反量化写入 vLLM paged KV。
  - 量化路径 **不会访问** `memory_obj.tensor`。

**约束：**
- 当前量化路径不支持 MLA 格式（`use_mla=True` 会抛错）。

## 4. C++/CUDA 反量化路径

### 4.1 PyBind 接口
位置：[csrc/pybind.cpp](csrc/pybind.cpp)

`multi_layer_kv_transfer` 同时支持两种输入：
- **Tensor**：非量化路径，保持原有行为。
- **TensorList**：量化路径，要求 `quantized=true` 且长度为 6。

### 4.2 C++ 接口与反量化实现
位置：[csrc/mem_kernels.cuh](csrc/mem_kernels.cuh)、[csrc/mem_kernels.cu](csrc/mem_kernels.cu)

**关键函数：**
- `multi_layer_kv_transfer(const std::vector<torch::Tensor>& ...)`：量化 TensorList 入口。
- `multi_layer_kv_transfer_dequantize(...)`：执行反量化并写入 vLLM paged KV。

**输入检查与约束：**
- `k_encoded/v_encoded` 必须是 int32，且连续。
- `k_scale/k_mn/v_scale/v_mn` 必须连续，且形状满足 KIVI 布局。
- `bits` ∈ {2,4,8}，`group_size` > 0。
- `slot_mapping` 为 CUDA tensor（int32 或 int64），长度必须满足：
  - `num_tokens == T_packed * (32 / bits)`。
- 不支持 `use_mla=true`。

**反量化内核行为：**
- kernel 根据 `slot_mapping` 定位 vLLM paged KV 的写入位置；
- 在 GPU 上逐 token/层/kv 执行反量化并写入。

### 4.3 预取/拷贝策略
在 `multi_layer_kv_transfer_dequantize()` 中：
- 若量化张量位于 pinned CPU，会在当前 CUDA stream 上异步拷贝到 GPU。
- 若量化张量已在 GPU，则直接使用。
- 这样避免了 Python 侧显式预取，简化调用链。

## 5. 数据流示意（量化分支）

```
CPU pinned (k_encoded/k_scale/k_mn/v_encoded/v_scale/v_mn)
        |
        |  (multi_layer_kv_transfer, quantized=True)
        v
CUDA op: multi_layer_kv_transfer_dequantize
        |
        |  dequantize + slot_mapping
        v
vLLM paged KV cache (GPU)
```

## 6. 关键约束与已知限制

- **MLA 不支持**：量化 retrieval 在 `use_mla=true` 下直接抛错。
- **TensorList 强约束**：必须是 6 张量，顺序固定。
- **slot_mapping 必须在 GPU**：否则内核拒绝执行。
- **精度由量化参数决定**：bits/group_size 必须与 store 时一致。

## 7. 相关文件索引

- 主流程： [lmcache/v1/cache_engine.py](lmcache/v1/cache_engine.py)
- GPU Connector： [lmcache/v1/gpu_connector.py](lmcache/v1/gpu_connector.py)
- PyBind 绑定： [csrc/pybind.cpp](csrc/pybind.cpp)
- CUDA/C++ 实现： [csrc/mem_kernels.cu](csrc/mem_kernels.cu)
- C++ 接口声明： [csrc/mem_kernels.cuh](csrc/mem_kernels.cuh)

## 8. 后续优化建议（可选）

- 增加 MLA 量化 retrieval 支持。
- 为 TensorList 路径增加更明确的 shape/dtype 自检信息。
- 提供针对量化路径的最小端到端测试用例。
