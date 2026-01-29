# LMCache KV Cache 量化开发日志

> **目标**: 为 LMCache 添加 KV cache 量化支持，在 `store()` 时量化，`retrieve()` 时反量化

---

## 环境信息
- **虚拟环境**: `/home/fei/research/llm/KVCache/LMCache-Quant/.venv`
- **包管理器**: uv
- **LMCache**: 已以 `-e` 模式安装
- **工作目录**: `/home/fei/research/llm/KVCache/LMCache-Quant-CC`
- **当前分支**: `kv-quant`

---

## 2024-01-28: 初始化分析

### 背景理解
1. **量化方法**: KIVI-style 量化（已存在于 `lmcache/v1/compute/quantization.py`）
   - `quantize_cache(cache, cache_type, group_size, bits)` → 返回 `(encoded, scale, mn)`
   - K cache: 沿 token 维度 (T) 分组量化
   - V cache: 沿 head_dim 维度 (D) 分组量化
   - 一个张量 → 三个张量 (encoded, scale, mn)

2. **内存管理机制** (`lmcache/v1/memory_management.py`)
   - `TensorMemoryObj` 已支持多张量（通过 `shapes`/`dtypes` list）
   - `get_tensor(index)` 可访问各个张量
   - `allocate(shapes, dtypes)` 支持不同 size/dtype 的张量列表

3. **Store 流程** (`lmcache/v1/cache_engine.py:313-506`)
   ```
   allocate MemoryObj → from_gpu (GPU→CPU) → batched_put
   ```

### 关键设计决策
- **量化位置**: 在 `from_gpu` 之后、`batched_put` 之前
- **head 处理**: 将所有 heads concat 作为 `nh=1` 处理，无需知道 num_heads
- **支持的 MemoryFormat**: KV_2LTD, KV_T2D, KV_2TD, KV_MLA_FMT

### 量化后的张量布局（6个张量）
```
|<----- k_encoded ---->|<-- k_scale -->|<-- k_mn -->|<----- v_encoded ---->|<-- v_scale -->|<-- v_mn -->|
```

---

## 2024-01-28: 添加配置参数

### 修改文件: `lmcache/v1/config.py`

#### 新增配置项 (_CONFIG_DEFINITIONS)
```python
"enable_kv_quantization": {
    "type": bool,
    "default": False,
    "description": "启用 KV cache 量化"
},
"kv_quantization_bits": {
    "type": int,
    "default": 4,
    "description": "量化位数 (2, 4, 8)"
},
"kv_quantization_group_size": {
    "type": int,
    "default": 128,
    "description": "量化组大小"
}
```

#### 配置验证 (_validate_config)
- `kv_quantization_bits` 必须是 2, 4, 或 8
- `kv_quantization_group_size` 必须大于 0

### 使用方式
```python
# 方式1: 配置文件
enable_kv_quantization: true
kv_quantization_bits: 4
kv_quantization_group_size: 128

# 方式2: 环境变量
export LMCACHE_ENABLE_KV_QUANTIZATION=true
export LMCACHE_KV_QUANTIZATION_BITS=4
export LMCACHE_KV_QUANTIZATION_GROUP_SIZE=128

# 方式3: 代码覆盖
config.enable_kv_quantization = True
config.kv_quantization_bits = 4
config.kv_quantization_group_size = 128
```

---

## 2024-01-29: CacheEngine Store 流程深入分析

### Store 流程详解

#### Step 1: MemoryObj 分配 (allocate)

**入口**: `cache_engine.py:421-426`

```python
kv_shapes = self.metadata.get_shapes(num_tokens)
kv_dtypes = self.metadata.get_dtypes()

memory_obj = self.storage_manager.allocate(
    kv_shapes,      # e.g., [torch.Size([2, 256, 128])]
    kv_dtypes,      # e.g., [torch.float16]
    busy_loop=self.force_store_wait,
    fmt=self.fmt,
)
```

**调用链**:
```
storage_manager.allocate()
    └── allocator_backend.allocate()
            └── TensorMemoryAllocator.allocate()
                    └── TensorMemoryObj.__init__()
```

**TensorMemoryAllocator.allocate** (`memory_management.py:1159-1203`):
- `shapes`/`dtypes` 支持单元素或多元素列表（多张量）
- 从预分配 buffer 中切片得到 `uint8` 的 `raw_data`
- `TensorMemoryObj.group_prefix_sum` 自动计算各张量偏移量

#### Step 2: From GPU (batched_from_gpu)

**入口**: `cache_engine.py:477-478`

```python
self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
```

**功能**: 将 GPU 上的 KV cache 数据传输到 CPU 的 MemoryObj 中

**关键点**: `memory_obj.tensor` 返回形状为 `meta.shape`、dtype 为 `meta.dtype` 的视图

#### Step 3: Batched Put (batched_put)

**入口**: `cache_engine.py:480-486`

**StorageManager.batched_put** (`storage_manager.py:388-437`):
1. 遍历所有 storage backends
2. 如果 backend 不同，调用 `allocate_and_copy_objects` 分配新内存并拷贝
3. 调用 `backend.batched_submit_put_task(ks, objs, transfer_spec)`
4. 对所有 memory_objs 调用 `ref_count_down()`

**LocalCPUBackend.batched_submit_put_task** (`local_cpu_backend.py:180-200`):
- 遍历调用 `submit_put_task(key, memory_obj, ...)`
- 在 `submit_put_task` 中：
  - `memory_obj.ref_count_up()` - 存入 hot_cache 前增加引用
  - `self.hot_cache[key] = memory_obj` - 存入缓存
  - 后续 `ref_count_down()` 会将引用从 2 减到 1

### 量化集成点分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CacheEngine.store()                           │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  ┌───────────┐           ┌──────────────┐           ┌─────────────┐
  │ allocate  │           │ batched_     │           │ batched_    │
  │ MemoryObj │──────────▶│ from_gpu     │──────────▶│ put         │
  └───────────┘           └──────────────┘           └─────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
  (分配空间)               (GPU→CPU 传输)            (存入 backend)

推荐量化位置: from_gpu 之后、batched_put 之前
```

**方案 A（在 from_gpu 之后、batched_put 之前）**:
- ✅ 优点: 量化数据更小，传输/存储带宽更小
- ❌ 缺点: CPU 需要做量化计算，会增加 store 时间

**方案 B（在 batched_put 时）**:
- ✅ 优点: 复用现有序列化接口
- ❌ 缺点: 需要修改更多地方

### 异步量化 + Put 设计

**参考**: `lmcache/v1/storage_backend/local_disk_backend.py`

LocalDiskBackend 使用 `AsyncPQThreadPoolExecutor` 实现异步存储：

```python
# LocalDiskBackend.submit_put_task
asyncio.run_coroutine_threadsafe(
    self.disk_worker.submit_task(
        "put",
        self.async_save_bytes_to_disk,
        key=key,
        memory_obj=memory_obj,
        on_complete_callback=on_complete_callback,
    ),
    self.loop,
)
```

**量化场景的异步设计**:

1. **修改 LocalCPUBackend.submit_put_task**:
   - 在 `submit_put_task` 中添加量化步骤
   - 将量化和保存一起包装成异步任务

2. **新增异步量化函数**:
```python
async def async_quantize_and_put(
    self,
    key: CacheEngineKey,
    ## 2026-01-29: Retrieve 侧 GPU 反量化（写入 vLLM paged KV）

    ### 目标与约束

    - **目标**: retrieve 时支持量化 KV 的反量化，并直接写入 vLLM paged KV cache。
    - **关键约束**:
        - **反量化发生在 GPU 上**（to_gpu 之后），避免 CPU 反量化带来的额外开销。
        - **复用现有 op**：不新增单独对外 Python API，仅扩展 `lmc_ops.multi_layer_kv_transfer`。
        - **保持 fast path**：允许量化张量停留在 pinned CPU（UVA 读取），避免显式 staging copy。

    ### 实现概要

    #### 1) Python 侧：把量化参数下传给 GPUConnector

    - 修改文件: `lmcache/v1/cache_engine.py`
    - 在 `retrieve` 流程中，当 `enable_kv_quantization=True` 时，将
        - `kv_quantization_bits`
        - `kv_quantization_group_size`
        通过 kwargs 下传给 `gpu_connector.batched_to_gpu(...)`。

    #### 2) Python 侧：VLLM paged connector 使用量化分支

    - 修改文件: `lmcache/v1/gpu_connector.py`
    - 在 `VLLMPagedMemGPUConnectorV2.to_gpu`：
        - 当 `memory_obj.metadata.is_quantized=True`，从多张量 `TensorMemoryObj` 取出 6 个张量：
            `(k_encoded, k_scale, k_mn, v_encoded, v_scale, v_mn)`。
        - 调用统一 op：
            `lmc_ops.multi_layer_kv_transfer(..., quantized=True, bits=..., group_size=...)`
        - CUDA kernel 在 GPU 上直接反量化并写入 vLLM paged KV。
        - MLA + quantized 路径暂未支持（显式 `NotImplementedError`）。

    #### 3) C++/CUDA 侧：扩展统一入口并新增 dequantize 写入逻辑

    - 修改文件: `csrc/pybind.cpp`
        - 保持兼容：`quantized=False` 时，参数 `key_value` 仍是原始 Tensor，走原逻辑。
        - 新增：`quantized=True` 时，`key_value` 为 6-tensor 序列，且仅支持 `direction=false` 且 `use_mla=false`。
        - 增加参数默认值：`quantized=false, bits=4, group_size=128`。

    - 修改文件: `csrc/mem_kernels.cu` / `csrc/mem_kernels.cuh`
        - 新增 `multi_layer_kv_transfer_dequantize(...)`：
            - 输入允许为 CUDA 或 pinned CPU（UVA 读取），输出为 vLLM paged KV。
            - 对参数/shape/dtype/contiguous 做了严格 `TORCH_CHECK`。
        - 兼容性修复：`slot_mapping` 支持 `int32` 和 `int64`（vLLM 里常见为 int32）。

    #### 4) Import 稳定性

    - 修改文件: `lmcache/__init__.py`
    - 提前 import torch，确保 `libc10.so` 等依赖先加载，避免某些环境下扩展导入失败。

    ### 测试

    - 新增文件: `tests/v1/test_quantized_kv_transfer.py`
        - 覆盖 `slot_mapping` 的 `torch.int32` / `torch.int64` 两种 dtype。
        - 验证 quantized path 的“GPU 反量化 + 写入 paged KV”与 Python 参考 `dequantize_cache` 一致（fp16 容差 `atol=2e-3`）。

    ### 安全性加固（Review follow-up）

    - CUDA kernel 增加 `slot_idx` 越界保护（`slot_idx >= page_buffer_size` 直接 return），避免潜在截断/OOB 写。
    - 非 vLLM paged connector（除 `VLLMPagedMemGPUConnectorV2` 外）在 `to_gpu/batched_to_gpu` 入口处遇到 `is_quantized=True` 会直接 `NotImplementedError`，防止 silent data corruption。

    memory_obj: MemoryObj,
    on_complete_callback: Callable[[CacheEngineKey], None] = None,
):
    # 1. 量化
    quantized_obj = quantize_kv_cache(memory_obj, bits, group_size)
    - [x] 在 `retrieve()` 中集成反量化（通过 GPUConnector 走量化分支）
    - [x] 根据 `is_quantized` 判断是否需要反量化
    - [x] 量化参数 (bits, group_size) 从 config 下传至 GPUConnector
    - [ ] MLA quantized retrieve 支持
        quantized_shapes, quantized_dtypes, fmt
    )

    # 3. 拷贝量化数据
    for i in range(6):
        new_obj.get_tensor(i).copy_(quantized_obj.get_tensor(i))

    # 4. 释放原 MemoryObj
    memory_obj.ref_count_down()

    # 5. 保存到 hot_cache (同步)
    self._do_submit_put(key, new_obj)

    # 6. 调用回调
    if on_complete_callback:
        on_complete_callback(key)
```

3. **在 CacheEngine.store() 中**:
```python
# 量化 + put 一起异步化，不阻塞主流程
if self.config.enable_kv_quantization:
    self._submit_quantized_put(keys, memory_objs, ...)
else:
    self.storage_manager.batched_put(keys, memory_objs, ...)
```

---

## 2024-01-29: 多张量 MemoryObj 可行性测试

### 测试结果

运行 `test_multi_tensor_memobj.py`，所有测试通过：

| 测试 | 结果 | 说明 |
|------|------|------|
| 单张量 MemoryObj | ✅ 通过 | 原始 KV cache 存储正常工作 |
| 多张量 MemoryObj | ✅ 通过 | 6 张量 MemoryObj 创建成功，get_tensor(i) 正常 |
| 量化流程 | ✅ 通过 | 量化→存储→反量化，误差 ~0.24（4-bit 正常） |
| 序列化 | ✅ 通过 | metadata 的 to_dict/from_dict 支持多张量 |
| 内存布局 | ✅ 通过 | 各张量内存不重叠，指针正确 |

### 关键发现

1. **现有 MemoryObj 结构完全兼容**:
   - `shapes`/`dtypes` 列表支持多张量
   - `get_tensor(index)` 正确返回各个张量
   - 内存布局通过 `group_prefix_sum` 自动计算

2. **量化流程可行**:
   - 原始 K cache: `[4, 256, 128]` float16
   - 量化后 6 个张量存储在单一 MemoryObj 中
   - 反量化后误差 ~0.24（4-bit 量化正常水平）

3. **无需修改现有代码**:
   - 只需在 store/retrieve 时机调用量化/反量化函数
   - 序列化已内置支持多张量

---

## 2024-01-29: 实现 Store Put 量化

### 修改文件: `lmcache/v1/storage_backend/storage_manager.py`

#### 1. 新增 imports (line 4)
```python
from concurrent.futures import Future, ThreadPoolExecutor
```

#### 2. 新增 `_quantize_memory_objects()` 函数 (lines 118-239)

参考 `allocate_and_copy_objects` 模式，在 StorageManager 模块级别实现：

```python
def _quantize_memory_objects(
    allocator_backend: AllocatorBackendInterface,
    memory_objs: list[MemoryObj],
    group_size: int,
    bits: int,
) -> list[MemoryObj]:
    """
    Quantize KV cache memory objects into multi-tensor format.
    K cache → (k_encoded, k_scale, k_mn)
    V cache → (v_encoded, v_scale, v_mn)
    """
```

**关键逻辑**:
- 调用 `quantize_cache()` 分别量化 K 和 V
- 计算 6 个张量的 shapes 和 dtypes
- 分配新内存存储量化数据
- 拷贝数据，释放原对象
- 返回量化后的 MemoryObj 列表

#### 3. StorageManager.__init__ 新增配置 (lines 385-395)

```python
# KV cache quantization config
self.enable_quantization = config.enable_kv_quantization
self.quantization_bits = config.kv_quantization_bits
self.quantization_group_size = config.kv_quantization_group_size

# Thread pool for CPU-bound quantization tasks (only if quantization enabled)
if self.enable_quantization:
    self.quantize_executor = ThreadPoolExecutor(max_workers=4)
else:
    self.quantize_executor = None
```

#### 4. batched_put() 修改 (lines 484-512)

```python
def batched_put(self, keys, memory_objs, transfer_spec=None, location=None):
    """
    If KV cache quantization is enabled, the memory objects will be
    quantized before storing.
    """
    if not memory_objs:
        return

    if self.enable_quantization:
        self._batched_put_with_quantization(keys, memory_objs, transfer_spec, location)
    else:
        self._batched_put_impl(keys, memory_objs, transfer_spec, location)
```

#### 5. 新增 `_batched_put_impl()` (lines 514-564)

原版 batched_put 逻辑重构为独立方法，添加 None 检查：
```python
def _batched_put_impl(self, keys, memory_objs, transfer_spec=None, location=None):
    """Original batched_put implementation."""
    # ... 原有逻辑 ...
    for memory_obj in objs:
        if memory_obj is not None:
            memory_obj.ref_count_down()
```

#### 6. 新增 `_batched_put_with_quantization()` (lines 566-620)

```python
def _batched_put_with_quantization(self, keys, memory_objs, transfer_spec=None, location=None):
    """
    Quantize and store KV cache asynchronously.
    """
    def quantize_and_put():
        # Step 1: 量化
        quantized_objs = _quantize_memory_objects(
            self.allocator_backend,
            memory_objs,
            self.quantization_group_size,
            self.quantization_bits,
        )

        # Step 2: batched_put
        self._batched_put_impl(valid_keys, valid_objs, transfer_spec, location)

    # 提交到显式 ThreadPoolExecutor (4 workers)
    self.loop.run_in_executor(self.quantize_executor, quantize_and_put)
```

#### 7. close() 新增 (lines 1299-1303)

```python
# Shutdown quantization executor
if self.quantize_executor is not None:
    self.quantize_executor.shutdown(wait=True)
```

### 异步执行模型

```
┌─────────────────────────────────────────────────────────────────┐
│                         整体线程模型                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ThreadPoolExecutor (4 workers)                          │
│  │  • quantize_and_put()  ← run_in_executor 提交           │
│  │    ├── 量化计算 (CPU-bound)                             │
│  │    └── _batched_put_impl()                              │
│  │          └── LocalDiskBackend.submit_put_task()         │
│  │                    └── run_coroutine_threadsafe()       │
│  │                              │                          │
│  │                              ▼                          │
│  │                    ┌─────────────────────┐             │
│  │                    │  Event Loop 线程    │             │
│  │                    │  async_save_bytes   │             │
│  │                    │  await 文件写入     │             │
│  │                    └─────────────────────┘             │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**线程说明**:
- **ThreadPoolExecutor (4 workers)**: 量化计算，CPU-bound
- **Event Loop 线程**: 异步 IO（磁盘、网络）

### 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 量化位置 | StorageManager.batched_put | 中间层，统一处理 |
| 线程池 | 显式 ThreadPoolExecutor(max_workers=4) | 更好控制，比默认更清晰 |
| 异步方式 | `loop.run_in_executor(executor, fn)` | CPU-bound 任务，不阻塞主流程 |
| 内存管理 | 量化后 ref_count_down 原对象 | 避免内存泄漏 |

### 测试结果

| 测试 | 结果 |
|------|------|
| 现有 memory_management 测试 (47个) | ✅ 全部通过 |
| 多张量 MemoryObj 测试 (5个) | ✅ 全部通过 |

### 待实现

- [x] 实现 Store Put 量化
- [ ] 实现 Retrieve 反量化
- [ ] 端到端测试

---

## 2024-01-29: KV 布局识别优化

### 问题背景

原实现直接用 `tensor[0]` / `tensor[1]` 提取 K/V cache，假设 shape 为 `[2, T, D]`。

但实际 KV cache 有多种布局：
- `KV_T2D`: `[2, T, D]` 或 `[2, L, T, D]` (L = num_layers)
- `KV_2LTD`: `[T, 2, D]` 或 `[L, 2, T, D]`
- `KV_MLA_FMT`: `[1, L, T, D]` 或 `[2, L, T, D]`

### 解决方案

基于 `MemoryFormat` 和 `shape` 识别布局，而非仅依赖 `tensor.dim()`：

```python
if fmt == MemoryFormat.KV_T2D:
    # [2, L, T, D] 或 [2, T, D]
    if shape[0] == 2:
        k_cache = tensor[0]  # [..., T, D]
        v_cache = tensor[1]  # [..., T, D]

elif fmt == MemoryFormat.KV_2LTD:
    # [L, 2, T, D] 或 [T, 2, D]
    if shape[1] == 2:
        k_cache = tensor[:, 0, :, :]  # [..., T, D]
        v_cache = tensor[:, 1, :, :]  # [..., T, D]

elif fmt == MemoryFormat.KV_MLA_FMT:
    # [1, L, T, D] 或 [2, L, T, D]
    if shape[0] == 1:
        k_cache = tensor[0]  # K-only
        v_cache = None
    elif shape[0] == 2:
        k_cache = tensor[0]
        v_cache = tensor[1]
```

**关键点**:
- L 维度（num_layers）直接作为 head 维度 `nh` 传给 `quantize_cache`
- `quantize_cache` 接受 `[nh, T, D]`，其中 `nh` 可以是任意值（包括 L）

---

## 2024-01-29: 新增 `is_quantized` 标记

### 问题背景

量化后的 MemoryObj 存储在后端，Retrieve 时需要知道是否需要反量化。

### 解决方案

在 `MemoryObjMetadata` 中添加 `is_quantized` 布尔字段：

```python
# lmcache/v1/memory_management.py

@dataclass
class MemoryObjMetadata:
    # ... 现有字段 ...

    # Whether the KV cache is quantized (for retrieval dequantization)
    is_quantized: bool = False
```

### 序列化/反序列化支持

```python
# to_dict() 添加
"is_quantized": self.is_quantized,

# from_dict() 读取（向后兼容，默认 False）
is_quantized = d.get("is_quantized", False)
```

### Store 时设置标记

```python
# storage_manager.py - _quantize_memory_objects()
quantized_obj.meta.is_quantized = True
```

### Retrieve 时判断

```python
memory_obj = backend.get_blocking(key)
if memory_obj.meta.is_quantized:
    # 反量化
    dequantized_obj = dequantize_memory_obj(memory_obj)
else:
    # 直接使用
    dequantized_obj = memory_obj
```

### 测试验证

| 测试 | 结果 |
|------|------|
| `is_quantized=True` 序列化/反序列化 | ✅ 通过 |
| 默认 `is_quantized=False` 向后兼容 | ✅ 通过 |

---

## 2024-01-29: Store Put 量化实现完成

### 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `lmcache/v1/config.py` | 新增 | 添加量化配置参数 |
| `lmcache/v1/memory_management.py` | 新增 | 添加 `is_quantized` 字段 |
| `lmcache/v1/storage_backend/storage_manager.py` | 新增 | 核心量化逻辑和异步存储 |
| `tests/v1/test_memory_management.py` | 新增 | `is_quantized` 序列化测试 |

---

### 1. lmcache/v1/config.py - 量化配置参数

#### 新增配置项

```python
"enable_kv_quantization": {
    "type": bool,
    "default": False,
    "env_converter": _to_bool,
    "description": "Enable KV cache quantization using KIVI-style quantization...",
},
"kv_quantization_bits": {
    "type": int,
    "default": 4,
    "env_converter": int,
    "description": "Number of bits for KV cache quantization (2, 4, or 8)...",
},
"kv_quantization_group_size": {
    "type": int,
    "default": 128,
    "env_converter": int,
    "description": "Group size for KV cache quantization...",
}
```

#### 配置验证

```python
if self.enable_kv_quantization:
    if self.kv_quantization_bits not in [2, 4, 8]:
        raise ValueError("kv_quantization_bits must be 2, 4, or 8")
    if self.kv_quantization_group_size <= 0:
        raise ValueError("kv_quantization_group_size must be positive")
```

---

### 2. lmcache/v1/memory_management.py - 量化标记

#### 新增字段

```python
@dataclass
class MemoryObjMetadata:
    # ... 现有字段 ...
    # Whether the KV cache is quantized (for retrieval dequantization)
    is_quantized: bool = False
```

#### 序列化支持

```python
# to_dict()
"is_quantized": self.is_quantized,

# from_dict()
is_quantized = d.get("is_quantized", False)  # 向后兼容，默认 False
```

---

### 3. lmcache/v1/storage_backend/storage_manager.py - 核心实现

#### 3.1 新增 `_quantize_memory_objects()` 函数

将 KV cache 量化为 6 个张量：

```
|<-- k_encoded (int32) -->|<-- k_scale -->|<-- k_mn -->|
|<-- v_encoded (int32) -->|<-- v_scale -->|<-- v_mn -->|
```

**KV 布局识别**:
- `KV_T2D`: `[2, T, D]` 或 `[2, L, T, D]` → `tensor[0]`, `tensor[1]`
- `KV_2LTD`: `[T, 2, D]` 或 `[L, 2, T, D]` → `tensor[:, 0, :]`, `tensor[:, 1, :]`
- `KV_MLA_FMT`: `[1/2, L, T, D]` → `tensor[0]`, `tensor[1]`（可选）

#### 3.2 修改 `allocate_and_copy_objects()` 函数

支持多张量 MemoryObj 的跨 backend 拷贝：
- 优先使用 `get_shapes()` / `get_dtypes()` 获取多张量信息
- 循环拷贝每个张量组 `get_tensor(i)`
- 同步 `is_quantized` 标记

#### 3.3 新增量化存储路径

```python
# StorageManager.__init__
if self.enable_quantization:
    self.quantize_executor = ThreadPoolExecutor(max_workers=4)

# batched_put() 分发
def batched_put(self, keys, memory_objs, ...):
    if self.enable_quantization:
        self._batched_put_with_quantization(keys, memory_objs, ...)
    else:
        self._batched_put_impl(keys, memory_objs, ...)

# 异步量化 + 存储
def _batched_put_with_quantization(self, keys, memory_objs, ...):
    def quantize_and_put():
        quantized_objs = _quantize_memory_objects(...)
        self._batched_put_impl(keys, quantized_objs, ...)
    self.loop.run_in_executor(self.quantize_executor, quantize_and_put)
```

#### 3.4 新增 `close()` 清理

```python
if self.quantize_executor is not None:
    self.quantize_executor.shutdown(wait=True)
```

---

### 4. 测试验证

| 测试场景 | 结果 |
|----------|------|
| KV_T2D `[2, T, D]` 量化 | ✅ 通过 |
| KV_T2D `[2, L, T, D]` 量化 | ✅ 通过 |
| KV_2LTD `[T, 2, D]` 量化 | ✅ 通过 |
| 4-bit 量化误差 (max ~0.24) | ✅ 通过 |
| `is_quantized` 序列化/反序列化 | ✅ 通过 |
| 向后兼容 (默认 `is_quantized=False`) | ✅ 通过 |

---

## 待实现

### Retrieve 反量化
- [ ] 在 `retrieve()` 中集成反量化
- [ ] 根据 `is_quantized` 判断是否需要反量化
- [ ] 量化参数 (bits, group_size) 需要持久化或从 config 读取

### 端到端测试
- [ ] 完整 store → retrieve 流程测试
- [ ] 量化压缩率验证
- [ ] 性能测试（量化 vs 非量化）
- [ ] 多 backend 兼容性测试

---

## 注意事项
1. **忽略预留接口**: `lmcache/v1/storage_backend/naive_serde/kivi_serde.py` 中的 `KIVISerializer` 是预留的 TODO，与本次实现无关
2. **兼容性**: 量化关闭时，系统应与原行为完全一致
3. **内存分配**: 量化后重新分配内存，原 MemoryObj 的 ref_count 需要正确处理
4. **Retrieve 依赖**: 当前 `is_quantized` 标记已就绪，但反量化逻辑待实现

---

## 参考资料
- KIVI 论文: https://arxiv.org/abs/2402.02750
- 量化实现: `lmcache/v1/compute/quantization.py`
- 内存管理: `lmcache/v1/memory_management.py`
- CacheEngine: `lmcache/v1/cache_engine.py`
