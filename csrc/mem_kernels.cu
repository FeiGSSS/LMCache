// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include "mem_kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#ifndef CHECK_CUDA_CALL
  #define CHECK_CUDA_CALL(call)                                             \
    do {                                                                    \
      cudaError_t err = call;                                               \
      if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        throw std::runtime_error(                                           \
            std::string("CUDA error in file '") + __FILE__ + "' in line " + \
            std::to_string(__LINE__) + " : " + cudaGetErrorString(err));    \
      }                                                                     \
    } while (0)
#endif

namespace lmc {

// inline helper to check MLA (callable from device and host)
__host__ __device__ __forceinline__ bool is_mla(
    const GPUKVFormat gpu_kv_format) {
  return gpu_kv_format == GPUKVFormat::NL_X_NB_BS_HS ||   // vllm MLA
         gpu_kv_format == GPUKVFormat::NL_X_NBBS_ONE_HS;  // SGLang MLA
}

template <typename scalar_t>
__global__ void load_and_reshape_flash_kernel(
    scalar_t* __restrict__ key_value,  // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ key_cache,    // [num_blocks, block_size,
                                               // num_heads, head_size]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, block_size,
                                               // num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_cache[src_key_value_idx];
    scalar_t tgt_value = value_cache[src_key_value_idx];

    key_value[tgt_key_idx] = tgt_key;
    key_value[tgt_value_idx] = tgt_value;
  }
}

template <typename scalar_t>
__global__ void reshape_and_cache_back_flash_kernel(
    const scalar_t* __restrict__ key_value,  // [num_tokens, num_heads,
                                             // head_size]
    scalar_t* __restrict__ key_cache,    // [num_blocks, block_size, num_heads,
                                         // head_size]
    scalar_t* __restrict__ value_cache,  // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_value[tgt_key_idx];
    scalar_t tgt_value = key_value[tgt_value_idx];

    key_cache[src_key_value_idx] = tgt_key;
    value_cache[src_key_value_idx] = tgt_value;
  }
}

template <typename scalar_t, bool USE_MLA>
__global__ void single_layer_kv_transfer_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,   // [num_tokens, 2,
                                                  // num_heads*head_size]
                                                  // or
                                                  // [2, num_tokens,
                                                  // num_heads*head_size]
                                                  // or for MLA:
                                                  // [num_tokens,
                                                  // aligned_head_size]
    scalar_t* __restrict__ vllm_key_value_cache,  // [2, num_blocks, block_size,
                                                  // num_heads, head_size] or
                                                  // [num_blocks, 2, block_size,
                                                  // num_heads, head_size]
                                                  // or for MLA:
                                                  // [num_blocks, block_size,
                                                  // head_size]

    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                                 block_offset * num_heads * head_size_in_64bit +
                                 head_idx * head_size_in_64bit + head_offset;

    if (direction == TransferDirection::D2H) {
      // GPU to LMCache
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      // For non-MLA, also copy the value component
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        lmc_key_value_cache[lmc_value_idx] =
            vllm_key_value_cache[vllm_value_idx];
      }
    } else {
      // LMCache to GPU
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      // For non-MLA, also copy the value component
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        vllm_key_value_cache[vllm_value_idx] =
            lmc_key_value_cache[lmc_value_idx];
      }
    }
  }
}

template <GPUKVFormat format>
__device__ __forceinline__ int64_t
page_buffer_offset(const int k_or_v, const int token_idx,
                   const int scalar_offset, const int scalars_per_token,
                   const int page_buffer_size, const int block_size) {
  // vllm cross layer
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash attention
  else if constexpr (format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash infer
  else if constexpr (format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    return block_idx * 2 * block_size * scalars_per_token +
           k_or_v * block_size * scalars_per_token +
           block_offset * scalars_per_token + scalar_offset;
  }
  // MLA formats: vLLM (NL_X_NB_BS_HS) and SGLang (NL_X_NBBS_ONE_HS)
  else if constexpr (format == GPUKVFormat::NL_X_NB_BS_HS ||
                     format == GPUKVFormat::NL_X_NBBS_ONE_HS) {
    return token_idx * scalars_per_token + scalar_offset;
  }
}

__device__ __forceinline__ int64_t page_buffer_offset_unilateral(
    const int token_idx, const int scalar_offset, const int scalars_per_token) {
  return token_idx * scalars_per_token + scalar_offset;
}

__device__ __forceinline__ int64_t
key_value_offset(const int k_or_v, const int layer_idx, const int token_idx,
                 const int scalar_offset, const int scalars_per_token,
                 const int num_tokens, const int num_layers) {
  return k_or_v * num_layers * num_tokens * scalars_per_token +
         layer_idx * num_tokens * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

template <typename scalar_t>
__global__ void single_layer_kv_transfer_sgl_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,  // [num_tokens, 2,
                                                 // num_heads*head_size]
                                                 // or
                                                 // [2, num_tokens,
                                                 // num_heads*head_size]
    scalar_t* __restrict__ sgl_key_cache,        // [num_blocks, block_size,
                                                 // num_heads, head_size]
    scalar_t* __restrict__ sgl_value_cache,      // [num_blocks, block_size,
                                                 // num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,    // [num_tokens]
    const int block_stride_in_64bit, const int lmc_stride,
    const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;
    const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t sgl_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    if (direction == TransferDirection::D2H) {
      lmc_key_value_cache[lmc_key_idx] = sgl_key_cache[sgl_key_value_idx];
      lmc_key_value_cache[lmc_value_idx] = sgl_value_cache[sgl_key_value_idx];
    } else {  // direction == TransferDirection::H2D
      sgl_key_cache[sgl_key_value_idx] = lmc_key_value_cache[lmc_key_idx];
      sgl_value_cache[sgl_key_value_idx] = lmc_key_value_cache[lmc_value_idx];
    }
  }
}

/**
 * Quickly load KV cache between vLLM paged memory and offloading buffer
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] <=> ptrs[block.y][block.z,
 * slot_id, thread.x]
 */
template <typename scalar_t, bool DIRECTION, GPUKVFormat format>
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers] * [2,
                                                // PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
                                                // or
                                                // [num_layers] * [num_blocks,
                                                // 2, block_size,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size, const int block_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t slot_idx = slot_mapping[token_id];
  scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t vllm_offset = page_buffer_offset<format>(
        k_or_v, slot_idx, i, scalars_per_token, page_buffer_size, block_size);

    if (DIRECTION)  // 1 is paged buffer to LMCache
      key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
    else  // 0 is LMCache to paged buffer
      paged_buffer_ptr[vllm_offset] = key_value[lmcache_offset];
  }
}

/*
 * handle sglang MHA offload between CPU and GPU
 * DIRECTION = 1 (true) means paged buffer to LMCache (D2H)
 * DIRECTION = 0 (false) means LMCache to paged buffer (H2D)
 */
template <typename scalar_t, bool DIRECTION>
__global__ void load_and_reshape_multi_layer_kernel_unilateral(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers *2] *
                                                // [PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t slot_idx = slot_mapping[token_id];
  scalar_t* key_ptr = paged_buffer_ptrs[layer_id];
  scalar_t* value_ptr = paged_buffer_ptrs[layer_id + num_layers];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t sgl_offset =
        page_buffer_offset_unilateral(slot_idx, i, scalars_per_token);

    if (k_or_v == 0) {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = key_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        key_ptr[sgl_offset] = key_value[lmcache_offset];
    } else {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = value_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        value_ptr[sgl_offset] = key_value[lmcache_offset];
    }
  }
}

}  // namespace lmc

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: H2D  means LMCache to PagedBuffer, D2H  means PagedBuffer to
 * LMCache
 */
#define LAUNCH_KERNEL_WITH_FORMAT(T, DIRECTION, FORMAT)                       \
  lmc::load_and_reshape_multi_layer_kernel<T, DIRECTION, FORMAT>              \
      <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,           \
                                   slot_mapping_ptr, num_xwords, num_tokens,  \
                                   num_layers, page_buffer_size, block_size); \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

template <typename T>
void multi_layer_kv_transfer_templated(
    torch::Tensor&
        key_value,  // key/value must be on gpu/pinned cpu.
                    // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn.
                    // [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA.
    const torch::Tensor& key_value_ptrs,  // [num_layers]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const TransferDirection direction, const GPUKVFormat gpu_kv_format,
    const int block_size) {
  T* key_value_ptr = get_kernel_ptr<T, torch::Tensor>(key_value);
  T** page_buffer_ptrs =
      get_kernel_ptr<T*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_xword = sizeof(T) / key_value.element_size();
  int num_xwords = num_origin_elements / elements_per_xword;

  int k_or_v_size = lmc::is_mla(gpu_kv_format) ? 1 : 2;

  dim3 grid(key_value.size(2), num_layers, k_or_v_size);
  dim3 block(std::min(num_xwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (direction == TransferDirection::H2D) {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        throw std::runtime_error("Unsupported GPUKVFormat");
    }
  } else {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        throw std::runtime_error("Unsupported GPUKVFormat");
    }
  }
}

#undef LAUNCH_KERNEL_WITH_FORMAT

/**
 * @see multi_layer_kv_transfer_templated
 */
void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format, const int block_size) {
  int num_origin_elements = key_value.size(3);
  int copy_size = num_origin_elements * key_value.element_size();
#ifndef LAUNCH_MULTI_LAYER_KV_TRANSFER
  #define LAUNCH_MULTI_LAYER_KV_TRANSFER(type)                          \
    do {                                                                \
      multi_layer_kv_transfer_templated<type>(                          \
          key_value, key_value_ptrs, slot_mapping, paged_memory_device, \
          page_buffer_size, direction, gpu_kv_format, block_size);      \
    } while (0)
#endif
  if (copy_size % 8 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int64_t);
  } else if (copy_size % 4 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int32_t);
  } else if (copy_size % 2 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int16_t);
  } else {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int8_t);
  }
#undef LAUNCH_MULTI_LAYER_KV_TRANSFER
}

/**
 * Quickly offload KV cache from SGLang paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in SGLang's K/V buffer has a shape of
 * [PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: H2D  means LMCache to PagedBuffer, D2H  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer_unilateral(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA key/value must be on gpu/pinned cpu

    const torch::Tensor& key_value_ptrs,  // [num_layers*2]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const TransferDirection direction, const GPUKVFormat gpu_kv_format) {
  const bool use_mla = lmc::is_mla(gpu_kv_format);
  // MLA case collapses back to multi_layer_kv_transfer
  // (vLLM and SGLang indexing are compatible)
  if (use_mla) {
    return multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping,
                                   paged_memory_device, page_buffer_size,
                                   direction, gpu_kv_format);
  }

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t** page_buffer_ptrs =
      get_kernel_ptr<int64_t*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_qword = 8 / key_value.element_size();
  int num_qwords = num_origin_elements / elements_per_qword;

  int k_or_v_size = 2;

  dim3 grid(key_value.size(2), key_value.size(1), k_or_v_size);
  dim3 block(std::min(num_qwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (direction == TransferDirection::H2D) {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, false>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, true>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void single_layer_kv_transfer(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
                                         // or for MLA:
                                         // [num_tokens, aligned_head_size]

    // torch::Tensor&
    //     vllm_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    // torch::Tensor&
    //     vllm_value_cache,  // [num_blocks, block_size, num_heads, head_size]
    //  key_cache/value_cache must be on gpu
    torch::Tensor&
        vllm_key_value_cache,  // [2, num_blocks, block_size, num_heads,
                               // head_size] for flash attention
    // [num_blocks, 2, block_size, num_heads, head_size] for flash infer
    // [num_blocks, block_size, head_size] for MLA

    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction, const GPUKVFormat gpu_kv_format,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);
  // int64_t* vllm_value_cache_ptr =
  //     get_kernel_ptr<int64_t, torch::Tensor>(vllm_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = lmc::is_mla(gpu_kv_format);

  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else {
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
    block_size = vllm_key_value_cache.size(2);
  }

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    // MLA format: [num_tokens, aligned_head_size]
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;  // No separate K/V for MLA
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;  // No separate K/V for MLA
  } else if (gpu_kv_format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {  // gpu_kv_format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  // int block_stride_in_64bit = vllm_key_cache.stride(0) / elements_per_entry;
  // TORCH_CHECK(vllm_key_cache.stride(0) == vllm_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch to the appropriate template specialization based on use_mla
  if (use_mla) {
    lmc::single_layer_kv_transfer_kernel<int64_t, true>
        <<<grid, block, 0, stream>>>(
            lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr,
            vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
            lmc_value_offset, num_heads, head_size_in_64bit, block_size,
            direction);
  } else {
    lmc::single_layer_kv_transfer_kernel<int64_t, false>
        <<<grid, block, 0, stream>>>(
            lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr,
            vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
            lmc_value_offset, num_heads, head_size_in_64bit, block_size,
            direction);
  }
}

void load_and_reshape_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens],
    const int layer_idx) {
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::load_and_reshape_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void reshape_and_cache_back_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const int layer_idx) {
  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::reshape_and_cache_back_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void single_layer_kv_transfer_sgl(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]

    torch::Tensor&
        sgl_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        sgl_value_cache,  // [num_blocks, block_size, num_heads, head_size]
                          // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* sgl_key_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_key_cache);
  int64_t* sgl_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / sgl_key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = sgl_key_cache.size(2);
  int head_size_in_64bit = sgl_key_cache.size(3) / elements_per_entry;

  int block_size = sgl_key_cache.size(1);

  int lmc_stride;
  int lmc_value_offset;
  if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int block_stride_in_64bit = sgl_key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(sgl_key_cache.stride(0) == sgl_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(sgl_key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::single_layer_kv_transfer_sgl_kernel<int64_t><<<grid, block, 0, stream>>>(
      lmc_key_value_cache_ptr, sgl_key_cache_ptr, sgl_value_cache_ptr,
      slot_mapping_ptr, block_stride_in_64bit, lmc_stride, lmc_value_offset,
      num_heads, head_size_in_64bit, block_size, direction);
}

/**
 * Perform asynchronous memory copy between lmcache host buffer (memory obj)
 * and a device buffer.
 * The copy will be performed asynchronously on the current CUDA stream.
 * They copy will be split into multiple smaller copies based on the host buffer
 * offset and host buffer alignment requirements.
 *
 * @param dest Destination pointer (device or host)
 * @param src Source pointer (device or host)
 * @param nbytes Number of bytes to copy
 * @param direction H2D or D2H
 * @param host_buffer_offset the virtual offset in the lmcache memory allocator
 * @param host_buffer_alignments the alignment (i.e., cudaHostRegister
 * granularity) requirement of the host buffer. Must be power of two.
 */
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  // Check that host_buffer_alignments is power of two
  TORCH_CHECK((host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  cudaMemcpyKind kind = (direction == TransferDirection::H2D)
                            ? cudaMemcpyHostToDevice
                            : cudaMemcpyDeviceToHost;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  while (offset < nbytes) {
    size_t current_src = src + offset;
    size_t current_dest = dest + offset;

    size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    size_t real_end = min(host_buffer_offset + nbytes, aligned_area_end);
    size_t max_nbytes = real_end - offset - host_buffer_offset;

    CHECK_CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(current_dest),
                                    reinterpret_cast<const void*>(current_src),
                                    max_nbytes, kind, stream));

    offset += max_nbytes;
  }
}

// ============================================================================
// Quantization support: GPU-side dequantization for KIVI-style quantized KV cache
// ============================================================================

namespace lmc {

// Legacy page_buffer_offset for dequantization kernel compatibility
__device__ __forceinline__ int64_t page_buffer_offset_legacy(
    const int k_or_v, const int token_idx, const int scalar_offset,
    const int scalars_per_token, const int page_buffer_size) {
  return k_or_v * page_buffer_size * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

__device__ __forceinline__ int unpack_int32_lowbit(const int32_t code,
                                                   const int bits,
                                                   const int idx_in_pack) {
  const int mask = (1 << bits) - 1;
  return (code >> (idx_in_pack * bits)) & mask;
}

template <typename scalar_t, typename index_t>
__global__ void dequantize_and_store_multi_layer_kernel_indexed(
    const int32_t* __restrict__ k_encoded,  // [L, T_packed, D]
    const scalar_t* __restrict__ k_scale,   // [L, T/group, 1, D]
    const scalar_t* __restrict__ k_mn,      // [L, T/group, 1, D]
    const int32_t* __restrict__ v_encoded,  // [L, T_packed, D]
    const scalar_t* __restrict__ v_scale,   // [L, T, D/group, 1]
    const scalar_t* __restrict__ v_mn,      // [L, T, D/group, 1]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers] * [2,
                                                // PAGE_BUFFER_SIZE, D]
    const index_t* __restrict__ slot_mapping,   // [num_tokens]
    const int T_packed, const int D, const int num_tokens,
    const int num_layers, const int page_buffer_size, const int bits,
    const int group_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;

  const int64_t slot_idx = static_cast<int64_t>(slot_mapping[token_id]);
  if (slot_idx < 0) {
    return;
  }
  // slot_idx is used as an int in page_buffer_offset; guard against
  // truncation and out-of-bounds writes.
  if (slot_idx >= static_cast<int64_t>(page_buffer_size)) {
    return;
  }

  const int feat_per_int = 32 / bits;
  const int packed_t = token_id / feat_per_int;
  const int idx_in_pack = token_id - packed_t * feat_per_int;

  scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const int32_t code =
        (k_or_v == 0)
            ? k_encoded[(layer_id * T_packed + packed_t) * D + d]
            : v_encoded[(layer_id * T_packed + packed_t) * D + d];
    const int q = unpack_int32_lowbit(code, bits, idx_in_pack);

    float scale_f;
    float mn_f;
    if (k_or_v == 0) {
      const int t_group = token_id / group_size;
      const int idx = (layer_id * (num_tokens / group_size) + t_group) * D + d;
      scale_f = static_cast<float>(k_scale[idx]);
      mn_f = static_cast<float>(k_mn[idx]);
    } else {
      const int d_group = d / group_size;
      const int v_groups = D / group_size;
      const int idx = (layer_id * num_tokens + token_id) * v_groups + d_group;
      scale_f = static_cast<float>(v_scale[idx]);
      mn_f = static_cast<float>(v_mn[idx]);
    }

    const float x = static_cast<float>(q) * scale_f + mn_f;
    const int64_t vllm_offset =
        page_buffer_offset_legacy(k_or_v, static_cast<int>(slot_idx), d, D,
                           page_buffer_size);
    paged_buffer_ptr[vllm_offset] = static_cast<scalar_t>(x);
  }
}

}  // namespace lmc

void multi_layer_kv_transfer_dequantize(
    const torch::Tensor& k_encoded, const torch::Tensor& k_scale,
    const torch::Tensor& k_mn, const torch::Tensor& v_encoded,
    const torch::Tensor& v_scale, const torch::Tensor& v_mn,
    const torch::Tensor& key_value_ptrs, const torch::Tensor& slot_mapping,
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const int bits, const int group_size) {
  // Allow encoded/scale/mn tensors to live on CUDA or pinned CPU.
  // This matches the existing multi_layer_kv_transfer behavior: kernels can
  // read from pinned host memory via UVA, avoiding an explicit staging copy.
  TORCH_CHECK(k_encoded.is_cuda() || (k_encoded.is_cpu() && k_encoded.is_pinned()),
              "k_encoded must be CUDA or pinned CPU");
  TORCH_CHECK(v_encoded.is_cuda() || (v_encoded.is_cpu() && v_encoded.is_pinned()),
              "v_encoded must be CUDA or pinned CPU");
  TORCH_CHECK(k_scale.is_cuda() || (k_scale.is_cpu() && k_scale.is_pinned()),
              "k_scale must be CUDA or pinned CPU");
  TORCH_CHECK(k_mn.is_cuda() || (k_mn.is_cpu() && k_mn.is_pinned()),
              "k_mn must be CUDA or pinned CPU");
  TORCH_CHECK(v_scale.is_cuda() || (v_scale.is_cpu() && v_scale.is_pinned()),
              "v_scale must be CUDA or pinned CPU");
  TORCH_CHECK(v_mn.is_cuda() || (v_mn.is_cpu() && v_mn.is_pinned()),
              "v_mn must be CUDA or pinned CPU");
  TORCH_CHECK(k_encoded.scalar_type() == at::kInt,
              "k_encoded must be int32");
  TORCH_CHECK(v_encoded.scalar_type() == at::kInt,
              "v_encoded must be int32");
  TORCH_CHECK(k_encoded.is_contiguous() && v_encoded.is_contiguous(),
              "encoded tensors must be contiguous");
  TORCH_CHECK(k_scale.is_contiguous() && k_mn.is_contiguous() &&
                  v_scale.is_contiguous() && v_mn.is_contiguous(),
              "scale/mn tensors must be contiguous");
  TORCH_CHECK(k_encoded.dim() == 3 && v_encoded.dim() == 3,
              "encoded tensors must be 3D [L, T_packed, D]");

  TORCH_CHECK(bits == 2 || bits == 4 || bits == 8,
              "bits must be one of {2,4,8}");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_buffer_size > 0, "page_buffer_size must be positive");
  TORCH_CHECK(page_buffer_size <= std::numeric_limits<int>::max(),
              "page_buffer_size must be <= INT_MAX");

  TORCH_CHECK(key_value_ptrs.is_cuda(), "key_value_ptrs must be a CUDA tensor");
  TORCH_CHECK(key_value_ptrs.scalar_type() == at::kLong,
              "key_value_ptrs must be int64 (torch.long)");
  TORCH_CHECK(key_value_ptrs.is_contiguous(), "key_value_ptrs must be contiguous");

  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong ||
                  slot_mapping.scalar_type() == at::kInt,
              "slot_mapping must be int64 (torch.long) or int32 (torch.int32)");

  const int num_layers = static_cast<int>(k_encoded.size(0));
  const int T_packed = static_cast<int>(k_encoded.size(1));
  const int D = static_cast<int>(k_encoded.size(2));
  const int feat_per_int = 32 / bits;
  const int num_tokens = static_cast<int>(slot_mapping.size(0));

  TORCH_CHECK(num_tokens == T_packed * feat_per_int,
              "slot_mapping length must equal T_packed*(32/bits)");
  TORCH_CHECK(num_tokens % group_size == 0,
              "num_tokens must be divisible by group_size for K dequant");
  TORCH_CHECK(D % group_size == 0,
              "D must be divisible by group_size for V dequant");

  const int expected_k_groups = num_tokens / group_size;
  TORCH_CHECK(k_scale.dim() == 4 && k_scale.size(0) == num_layers &&
                  k_scale.size(1) == expected_k_groups && k_scale.size(2) == 1 &&
                  k_scale.size(3) == D,
              "k_scale must be [L, T/group, 1, D]");
  TORCH_CHECK(k_mn.sizes() == k_scale.sizes(), "k_mn must match k_scale");

  const int expected_v_groups = D / group_size;
  TORCH_CHECK(v_scale.dim() == 4 && v_scale.size(0) == num_layers &&
                  v_scale.size(1) == num_tokens &&
                  v_scale.size(2) == expected_v_groups && v_scale.size(3) == 1,
              "v_scale must be [L, T, D/group, 1]");
  TORCH_CHECK(v_mn.sizes() == v_scale.sizes(), "v_mn must match v_scale");

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Prefetch quantized tensors to GPU if they are on pinned CPU.
  // Copies are enqueued on the current stream to preserve ordering.
  torch::Tensor k_encoded_dev = k_encoded;
  torch::Tensor v_encoded_dev = v_encoded;
  torch::Tensor k_scale_dev = k_scale;
  torch::Tensor k_mn_dev = k_mn;
  torch::Tensor v_scale_dev = v_scale;
  torch::Tensor v_mn_dev = v_mn;

  if (!k_encoded.is_cuda()) {
    k_encoded_dev = k_encoded.to(paged_memory_device, k_encoded.scalar_type(),
                                 /*non_blocking=*/true, /*copy=*/true);
  }
  if (!v_encoded.is_cuda()) {
    v_encoded_dev = v_encoded.to(paged_memory_device, v_encoded.scalar_type(),
                                 /*non_blocking=*/true, /*copy=*/true);
  }
  if (!k_scale.is_cuda()) {
    k_scale_dev = k_scale.to(paged_memory_device, k_scale.scalar_type(),
                             /*non_blocking=*/true, /*copy=*/true);
  }
  if (!k_mn.is_cuda()) {
    k_mn_dev = k_mn.to(paged_memory_device, k_mn.scalar_type(),
                       /*non_blocking=*/true, /*copy=*/true);
  }
  if (!v_scale.is_cuda()) {
    v_scale_dev = v_scale.to(paged_memory_device, v_scale.scalar_type(),
                             /*non_blocking=*/true, /*copy=*/true);
  }
  if (!v_mn.is_cuda()) {
    v_mn_dev = v_mn.to(paged_memory_device, v_mn.scalar_type(),
                       /*non_blocking=*/true, /*copy=*/true);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, k_scale_dev.scalar_type(),
      "multi_layer_kv_transfer_dequantize", [&] {
        const int32_t* k_encoded_ptr =
            get_kernel_ptr<const int32_t, const torch::Tensor>(k_encoded_dev);
        const int32_t* v_encoded_ptr =
            get_kernel_ptr<const int32_t, const torch::Tensor>(v_encoded_dev);
        const scalar_t* k_scale_ptr =
            get_kernel_ptr<const scalar_t, const torch::Tensor>(k_scale_dev);
        const scalar_t* k_mn_ptr =
            get_kernel_ptr<const scalar_t, const torch::Tensor>(k_mn_dev);
        const scalar_t* v_scale_ptr =
            get_kernel_ptr<const scalar_t, const torch::Tensor>(v_scale_dev);
        const scalar_t* v_mn_ptr =
            get_kernel_ptr<const scalar_t, const torch::Tensor>(v_mn_dev);
        scalar_t** page_buffer_ptrs =
            get_kernel_ptr<scalar_t*, const torch::Tensor>(key_value_ptrs);

        dim3 grid(num_tokens, num_layers, 2);
        dim3 block(256);

        if (slot_mapping.scalar_type() == at::kLong) {
          const int64_t* slot_mapping_ptr =
            get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
          lmc::dequantize_and_store_multi_layer_kernel_indexed<scalar_t, int64_t>
            <<<grid, block, 0, stream>>>(
              k_encoded_ptr, k_scale_ptr, k_mn_ptr, v_encoded_ptr,
              v_scale_ptr, v_mn_ptr, page_buffer_ptrs, slot_mapping_ptr,
              T_packed, D, num_tokens, num_layers, page_buffer_size, bits,
              group_size);
        } else {
          const int32_t* slot_mapping_ptr =
            get_kernel_ptr<const int32_t, const torch::Tensor>(slot_mapping);
          lmc::dequantize_and_store_multi_layer_kernel_indexed<scalar_t, int32_t>
            <<<grid, block, 0, stream>>>(
              k_encoded_ptr, k_scale_ptr, k_mn_ptr, v_encoded_ptr,
              v_scale_ptr, v_mn_ptr, page_buffer_ptrs, slot_mapping_ptr,
              T_packed, D, num_tokens, num_layers, page_buffer_size, bits,
              group_size);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

// Overload: accept a list of tensors for quantized KV transfer.
// Expects 6 tensors: (k_encoded, k_scale, k_mn, v_encoded, v_scale, v_mn).
// Delegates to multi_layer_kv_transfer_dequantize for GPU-side dequantization.
void multi_layer_kv_transfer(const std::vector<torch::Tensor>& key_value_list,
               const torch::Tensor& key_value_ptrs,
               const torch::Tensor& slot_mapping,
               const torch::Device& paged_memory_device,
               const int page_buffer_size, const bool direction,
               const bool use_mla, const int bits,
               const int group_size) {
  TORCH_CHECK(!direction,
        "TensorList path supports only LMCache->vLLM (direction=false)");
  TORCH_CHECK(!use_mla, "TensorList path does not support MLA format yet");
  TORCH_CHECK(key_value_list.size() == 6,
        "TensorList expects 6 tensors: "
        "(k_encoded, k_scale, k_mn, v_encoded, v_scale, v_mn)");

  return multi_layer_kv_transfer_dequantize(
    key_value_list[0], key_value_list[1], key_value_list[2],
    key_value_list[3], key_value_list[4], key_value_list[5],
    key_value_ptrs, slot_mapping, paged_memory_device, page_buffer_size,
    bits, group_size);
}
