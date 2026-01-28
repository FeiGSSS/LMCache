# SPDX-License-Identifier: Apache-2.0
"""
KIVI Quantization Module

Implements KIVI-style quantization for KV cache following the original paper:
- K cache: Group-wise quantization along token dimension (T)
- V cache: Group-wise quantization along head_dim dimension (D)

Reference:
    KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache
    https://arxiv.org/abs/2402.02750
"""

from typing import Tuple
import torch


def pack_tensor(data: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """
    Pack low-bit quantized values into int32 tensors.

    Args:
        data: Quantized int32 tensor with values in [0, 2^bits-1]
        bits: Number of bits per value (2, 4, or 8)
        pack_dim: Dimension to pack along

    Returns:
        Packed int32 tensor where pack_dim size is reduced by factor of (32//bits)

    Example:
        >>> data = torch.randint(0, 16, (2, 32, 256, 128))  # [B, nh, T, D]
        >>> packed = pack_tensor(data, bits=4, pack_dim=2)
        >>> packed.shape  # [2, 32, 32, 128]  (T/8)
    """
    assert bits in [2, 4, 8], f"Only 2, 4, 8 bits are supported, got {bits}"

    # Handle negative pack_dim
    ndim = data.dim()
    pack_dim = (pack_dim + ndim) % ndim

    shape = data.shape
    feat_per_int = 32 // bits
    assert shape[pack_dim] % feat_per_int == 0, (
        f"Dimension {pack_dim} (size={shape[pack_dim]}) must be "
        f"divisible by {feat_per_int} for {bits}-bit packing"
    )

    # 1. Reshape to [..., dim // feat, feat, ...]
    # We split the packing dimension into two: (groups, group_size)
    new_shape = list(shape)
    new_shape[pack_dim] //= feat_per_int
    new_shape.insert(pack_dim + 1, feat_per_int)

    data_view = data.contiguous().view(new_shape)

    # 2. Create shifts tensor: [0, bits, 2*bits, ...]
    shifts = torch.arange(0, 32, bits, device=data.device, dtype=torch.int32)

    # 3. Reshape shifts for broadcasting: [1, ..., 1, feat, 1, ..., 1]
    # The 'feat' dimension is at pack_dim + 1 in data_view
    shift_shape = [1] * (ndim + 1)
    shift_shape[pack_dim + 1] = feat_per_int
    shifts = shifts.view(shift_shape)

    # 4. Pack: Shift and sum (equivalent to bitwise OR for non-overlapping bits)
    # Reducing along the 'feat' dimension (pack_dim + 1)
    code = (data_view << shifts).sum(dim=pack_dim + 1)

    return code.to(torch.int32)


def unpack_tensor(code: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    """
    Unpack low-bit values from int32 tensors.

    Args:
        code: Packed int32 tensor
        bits: Number of bits per value (2, 4, or 8)
        pack_dim: Dimension that was packed along

    Returns:
        Unpacked tensor where pack_dim size is multiplied by (32//bits).
        Returns int8 for 2/4-bit (values in [-128, 127] or [0, 15] range),
        uint8 for 8-bit (values in [0, 255] range).
        This compact storage is efficient; dequantize will convert to target dtype.

    Example:
        >>> packed = torch.randint(0, 2**32, (2, 32, 32, 128), dtype=torch.int32)
        >>> unpacked = unpack_tensor(packed, bits=4, pack_dim=2)
        >>> unpacked.shape  # [2, 32, 256, 128]  (T*8)
        >>> unpacked.dtype  # torch.int8 (compact storage)
    """
    assert bits in [2, 4, 8], f"Only 2, 4, 8 bits are supported, got {bits}"

    # Handle negative pack_dim
    ndim = code.dim()
    pack_dim = (pack_dim + ndim) % ndim

    feat_per_int = 32 // bits
    mask = (1 << bits) - 1

    # 1. Prepare shifts: [0, bits, 2*bits, ...]
    # Shape logic: we want to broadcast shifts against a new dimension inserted at pack_dim + 1
    shifts = torch.arange(0, 32, bits, device=code.device, dtype=torch.int32)

    # Reshape shifts to [1, ..., 1, feat, 1, ..., 1]
    shift_shape = [1] * (ndim + 1)
    shift_shape[pack_dim + 1] = feat_per_int
    shifts = shifts.view(shift_shape)

    # 2. Expand code to prepare for broadcasting
    # [..., dim, ...] -> [..., dim, 1, ...]
    code_expanded = code.unsqueeze(pack_dim + 1)

    # 3. Unpack using broadcasting
    # (code >> shifts) handles the bit shifting for all packed elements at once
    # & mask extracts the relevant bits
    unpacked_vals = (code_expanded >> shifts) & mask

    # 4. Cast to appropriate type
    # For 8-bit, we use uint8 to represent 0-255 correctly.
    # For 2/4-bit, values fit in int8.
    output_dtype = torch.uint8 if bits == 8 else torch.int8
    unpacked_vals = unpacked_vals.to(output_dtype)

    # 5. Merge the dimensions: [..., dim, feat, ...] -> [..., dim * feat, ...]
    out_shape = list(code.shape)
    out_shape[pack_dim] *= feat_per_int

    return unpacked_vals.view(out_shape)


def quantize_cache(
    cache: torch.Tensor,
    cache_type: str,  # 'k' for K cache, 'v' for V cache
    group_size: int,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize KV cache following KIVI scheme with unified packing along T dimension.

    For K cache: group-wise along token dimension (T)
    For V cache: group-wise along head_dim dimension (D)

    Args:
        cache: KV cache tensor with shape [nh, T, D] (any float dtype: fp16, fp32, bf16)
        cache_type: Cache type - 'k' for K cache, 'v' for V cache
        group_size: Group size for quantization
        bits: Quantization bits (2, 4, or 8)

    Returns:
        Tuple of (encoded, scale, mn):
        - encoded: Packed quantized values (int32), shape [nh, T/8, D] for 4-bit
        - scale: Quantization scale (same dtype as input cache)
        - mn: Quantization min (same dtype as input cache)

    Example:
        >>> k = torch.randn(32, 256, 128, dtype=torch.float16)  # [nh, T, D]
        >>> encoded, scale, mn = quantize_cache(k, 'k', group_size=32, bits=4)
        >>> encoded.shape  # [32, 32, 128]  (T/8)
        >>> scale.dtype  # torch.float16 (same as input)
    """
    assert len(cache.shape) == 3, f"Expected 3D tensor [nh, T, D], got shape {cache.shape}"
    assert cache_type in ['k', 'v'], f"cache_type must be 'k' or 'v', got {cache_type}"
    nh, T, D = cache.shape

    if cache_type == 'k':
        # K cache: group along T dimension
        assert T % group_size == 0, (
            f"T dimension ({T}) must be divisible by group_size ({group_size})"
        )
        num_groups = T // group_size
        # Reshape to [nh, num_groups, group_size, D]
        new_shape = (nh, num_groups, group_size, D)
        dim_to_reduce = -2  # Reduce along group_size dimension
    else:  # cache_type == 'v'
        # V cache: group along D dimension
        assert D % group_size == 0, (
            f"D dimension ({D}) must be divisible by group_size ({group_size})"
        )
        num_groups = D // group_size
        # Reshape to [nh, T, num_groups, group_size]
        new_shape = (nh, T, num_groups, group_size)
        dim_to_reduce = -1  # Reduce along group_size dimension

    max_int = 2**bits - 1

    data = cache.view(new_shape)

    # Compute min/max along group_size dimension
    mn = torch.min(data, dim=dim_to_reduce, keepdim=True)[0]
    mx = torch.max(data, dim=dim_to_reduce, keepdim=True)[0]
    scale = (mx - mn) / max_int

    # Numerical stability: prevent division by zero
    # When all values in a group are identical (e.g., padding tokens),
    # mx == mn, resulting in scale == 0. Add small epsilon to avoid NaN/Inf.
    # Use max of absolute values as reference for relative epsilon
    range_max = torch.maximum(mx.abs(), mn.abs())
    epsilon = torch.finfo(scale.dtype).eps * torch.clamp(range_max, min=1.0)
    scale = torch.maximum(scale, epsilon)

    # Quantize: q = round((x - mn) / scale)
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)

    # Reshape back to [nh, T, D]
    data = data.view([nh, T, D])

    # Pack along T dimension (unified for both K and V, pack_dim=1 for 3D tensor)
    # For 3D tensor [nh, T, D], T is at index 1
    code = pack_tensor(data, bits, pack_dim=1)

    return code, scale, mn


def dequantize_cache(
    encoded: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    cache_type: str,  # 'k' for K cache, 'v' for V cache
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """
    Dequantize KV cache with unified unpacking along T dimension.

    For K cache: group-wise along token dimension (T)
    For V cache: group-wise along head_dim dimension (D)

    Args:
        encoded: Packed quantized values, shape [nh, T/feat_per_int, D]
        scale: Quantization scale (dtype determines output dtype)
        mn: Quantization min, same shape and dtype as scale
        cache_type: Cache type - 'k' for K cache, 'v' for V cache
        group_size: Group size used for quantization
        bits: Quantization bits (2, 4, or 8)

    Returns:
        Dequantized tensor with shape [nh, T, D] and dtype matching scale/mn
    """
    assert cache_type in ['k', 'v'], f"cache_type must be 'k' or 'v', got {cache_type}"

    # Recover original shape from encoded shape
    # encoded.shape = [nh, T/feat_per_int, D]
    # original shape = [nh, T, D] = [nh, T/feat_per_int * feat_per_int, D]
    feat_per_int = 32 // bits
    nh, T_packed, D = encoded.shape
    T = T_packed * feat_per_int

    # Get the original dtype from scale/mn (preserved from quantization)
    dtype = scale.dtype

    # Step 1: Unpack (unified pack_dim=1 for both K and V)
    # unpack_tensor returns int8/uint8 for memory efficiency
    data = unpack_tensor(encoded, bits, pack_dim=1)  # [nh, T, D]

    # Step 2: Reshape to grouped format
    if cache_type == 'k':
        num_groups = T // group_size
        # Reshape to [nh, num_groups, group_size, D]
        data = data.view([nh, num_groups, group_size, D])
    else:  # cache_type == 'v'
        num_groups = D // group_size
        # Reshape to [nh, T, num_groups, group_size]
        data = data.view([nh, T, num_groups, group_size])

    # Step 3: Dequantize: x = q * scale + mn
    # Convert to target dtype before arithmetic for precision.
    # PyTorch would auto-promote int8 to float during multiplication,
    # but explicit conversion ensures we use the exact target dtype.
    data = data.to(dtype)
    data = data * scale + mn

    # Step 4: Reshape back to original shape
    return data.view([nh, T, D])
