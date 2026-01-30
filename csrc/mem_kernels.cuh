// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

// #ifndef MEM_KERNELS_CUH
// #define MEM_KERNELS_CUH

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

void multi_layer_kv_transfer(torch::Tensor& key_value,
                             const torch::Tensor& key_value_ptrs,
                             const torch::Tensor& slot_mapping,
                             const torch::Device& paged_memory_device,
                             const int page_buffer_size, const bool direction,
                             const bool use_mla);

// Overload: accept a list of tensors for quantized KV transfer.
// Expects 6 tensors: (k_encoded, k_scale, k_mn, v_encoded, v_scale, v_mn).
void multi_layer_kv_transfer(const std::vector<torch::Tensor>& key_value_list,
                             const torch::Tensor& key_value_ptrs,
                             const torch::Tensor& slot_mapping,
                             const torch::Device& paged_memory_device,
                             const int page_buffer_size, const bool direction,
                             const bool use_mla, const int bits,
                             const int group_size);

void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const bool direction, const bool use_mla);

// Dequantize quantized KV cache and write directly into vLLM paged memory.
// Quantization format is KIVI-style produced by lmcache.v1.compute.quantization.quantize_cache:
// - k_encoded/v_encoded: int32, shape [L, T_packed, D]
// - k_scale/k_mn: float16/bf16, shape [L, T/group_size, 1, D]
// - v_scale/v_mn: float16/bf16, shape [L, T, D/group_size, 1]
// key_value_ptrs points to per-layer vLLM KV tensors of shape [2, PAGE_BUFFER_SIZE, D].
void multi_layer_kv_transfer_dequantize(
  const torch::Tensor& k_encoded,
  const torch::Tensor& k_scale,
  const torch::Tensor& k_mn,
  const torch::Tensor& v_encoded,
  const torch::Tensor& v_scale,
  const torch::Tensor& v_mn,
  const torch::Tensor& key_value_ptrs,
  const torch::Tensor& slot_mapping,
  const torch::Device& paged_memory_device,
  const int page_buffer_size,
  const int bits,
  const int group_size);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping, const bool direction,
                              const bool token_major = false,
                              const bool vllm_two_major = false,
                              const bool use_mla = false);

void single_layer_kv_transfer_sgl(torch::Tensor& lmc_key_value_cache,
                                  torch::Tensor& sgl_key_cache,
                                  torch::Tensor& sgl_value_cache,
                                  torch::Tensor& slot_mapping,
                                  const bool direction,
                                  const bool token_major = false);

void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);
