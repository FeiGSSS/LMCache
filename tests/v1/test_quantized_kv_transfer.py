# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lmcache import c_ops as lmc_ops
from lmcache.v1.compute.quantization import dequantize_cache, quantize_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("slot_mapping_dtype", [torch.int32, torch.int64])
def test_multi_layer_kv_transfer_quantized_dequant_to_vllm(slot_mapping_dtype: torch.dtype):
    # Keep dimensions small but aligned with constraints:
    # - num_tokens divisible by group_size
    # - D divisible by group_size
    # - num_tokens == T_packed * (32/bits)
    device = torch.device("cuda")

    bits = 4
    group_size = 128
    num_layers = 2
    num_tokens = 256
    D = 128
    page_buffer_size = 1024

    # Original KV cache: [L, T, D]
    k = torch.randn((num_layers, num_tokens, D), device=device, dtype=torch.float16)
    v = torch.randn((num_layers, num_tokens, D), device=device, dtype=torch.float16)

    # Quantize (K grouped on T, V grouped on D)
    k_encoded, k_scale, k_mn = quantize_cache(k, "k", group_size=group_size, bits=bits)
    v_encoded, v_scale, v_mn = quantize_cache(v, "v", group_size=group_size, bits=bits)

    # Allocate a simple "vLLM paged" buffer per layer: [2, PAGE_BUFFER_SIZE, D]
    paged_layers = [
        torch.empty((2, page_buffer_size, D), device=device, dtype=torch.float16)
        for _ in range(num_layers)
    ]

    # Pointer array: CPU pinned -> CUDA (matches connector behavior)
    ptrs_cpu = torch.empty((num_layers,), device="cpu", dtype=torch.int64, pin_memory=True)
    ptrs_cpu.numpy()[:] = [t.data_ptr() for t in paged_layers]
    key_value_ptrs = ptrs_cpu.to(device=device)

    # Slot mapping (int32 or int64). Use a simple injective mapping into [0, page_buffer_size).
    slot_mapping = (torch.arange(num_tokens, device=device) % page_buffer_size).to(
        dtype=slot_mapping_dtype
    )

    # Invoke unified op: dequantize on GPU and write into paged KV cache.
    lmc_ops.multi_layer_kv_transfer(
        [k_encoded, k_scale, k_mn, v_encoded, v_scale, v_mn],
        key_value_ptrs,
        slot_mapping,
        device,
        page_buffer_size,
        lmc_ops.TransferDirection.H2D,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        0,  # block_size (not used for NL_X_TWO_NB_BS_NH_HS format)
        bits,
        group_size,
    )

    # Reference: dequantize on GPU then compare the specific slots that were written.
    k_ref = dequantize_cache(k_encoded, k_scale, k_mn, "k", group_size=group_size, bits=bits)
    v_ref = dequantize_cache(v_encoded, v_scale, v_mn, "v", group_size=group_size, bits=bits)

    # Validate a handful of tokens to keep test fast.
    check_tokens = torch.tensor([0, 1, 7, 8, 63, 127, 128, 255], device=device)
    check_slots = slot_mapping[check_tokens].to(torch.int64)

    for layer_id in range(num_layers):
        got_k = paged_layers[layer_id][0, check_slots, :]
        got_v = paged_layers[layer_id][1, check_slots, :]
        exp_k = k_ref[layer_id, check_tokens, :]
        exp_v = v_ref[layer_id, check_tokens, :]

        # CUDA kernel dequantizes in float32 then casts to fp16/bf16,
        # while the Python reference may use lower-precision intermediates.
        # Allow a small atol consistent with fp16 quantization error.
        torch.testing.assert_close(got_k, exp_k, rtol=0.0, atol=4e-3)
        torch.testing.assert_close(got_v, exp_v, rtol=0.0, atol=4e-3)
