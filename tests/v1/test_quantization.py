# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for KIVI quantization module.
"""

import pytest
import torch

from lmcache.v1.compute.quantization import (
    pack_tensor,
    unpack_tensor,
    quantize_cache,
    dequantize_cache,
)


class TestPackUnpack:
    """Test pack_tensor and unpack_tensor functions."""

    def test_pack_unpack_4bit_pack_dim_1(self):
        """Test 4-bit pack/unpack along dimension 1 (3D tensor)."""
        data = torch.randint(0, 16, (4, 256, 128), dtype=torch.int32)  # [nh, T, D]
        packed = pack_tensor(data, bits=4, pack_dim=1)
        assert packed.shape == (4, 32, 128)  # 256 / 8 = 32

        unpacked = unpack_tensor(packed, bits=4, pack_dim=1)
        assert unpacked.shape == data.shape
        assert torch.equal(unpacked.to(torch.int32), data)

    def test_pack_unpack_4bit_pack_dim_2(self):
        """Test 4-bit pack/unpack along dimension 2 (4D tensor for backward compatibility)."""
        data = torch.randint(0, 16, (2, 4, 256, 128), dtype=torch.int32)
        packed = pack_tensor(data, bits=4, pack_dim=2)
        assert packed.shape == (2, 4, 32, 128)  # 256 / 8 = 32

        unpacked = unpack_tensor(packed, bits=4, pack_dim=2)
        assert unpacked.shape == data.shape
        assert torch.equal(unpacked.to(torch.int32), data)

    def test_pack_unpack_2bit_pack_dim_1(self):
        """Test 2-bit pack/unpack along dimension 1 (3D tensor)."""
        data = torch.randint(0, 4, (4, 256, 128), dtype=torch.int32)  # [nh, T, D]
        packed = pack_tensor(data, bits=2, pack_dim=1)
        assert packed.shape == (4, 16, 128)  # 256 / 16 = 16

        unpacked = unpack_tensor(packed, bits=2, pack_dim=1)
        assert unpacked.shape == data.shape
        assert torch.equal(unpacked.to(torch.int32), data)

    def test_pack_unpack_8bit_pack_dim_1(self):
        """Test 8-bit pack/unpack along dimension 1 (3D tensor)."""
        data = torch.randint(0, 256, (4, 256, 128), dtype=torch.int32)  # [nh, T, D]
        packed = pack_tensor(data, bits=8, pack_dim=1)
        assert packed.shape == (4, 64, 128)  # 256 / 4 = 64

        unpacked = unpack_tensor(packed, bits=8, pack_dim=1)
        assert unpacked.shape == data.shape
        assert torch.equal(unpacked.to(torch.int32), data)

    def test_pack_invalid_bits(self):
        """Test pack_tensor with invalid bits."""
        data = torch.randint(0, 16, (4, 256, 128), dtype=torch.int32)
        with pytest.raises(AssertionError):
            pack_tensor(data, bits=3, pack_dim=1)

    def test_pack_invalid_dimension(self):
        """Test pack_tensor with dimension not divisible by feat_per_int."""
        data = torch.randint(0, 16, (4, 255, 128), dtype=torch.int32)  # 255 not divisible by 8
        with pytest.raises(AssertionError):
            pack_tensor(data, bits=4, pack_dim=1)


class TestQuantizeKCache:
    """Test K cache quantization and dequantization."""

    @pytest.mark.parametrize("group_size,bits", [
        (32, 4),
        (64, 4),
        (32, 2),
        (32, 8),
    ])
    def test_quantize_dequantize_k_cache(self, group_size, bits):
        """Test K cache quantization and dequantization."""
        k_cache = torch.randn(32, group_size * 8, 128, dtype=torch.float16)  # [nh, T, D]
        original_shape = k_cache.shape

        # Quantize
        encoded, scale, mn = quantize_cache(k_cache, 'k', group_size, bits)

        # Check shapes
        feat_per_int = 32 // bits
        assert encoded.shape == (32, (group_size * 8) // feat_per_int, 128)  # [nh, T/feat, D]
        assert scale.shape == (32, 8, 1, 128)  # [nh, num_groups, 1, D]
        assert mn.shape == (32, 8, 1, 128)

        # Dequantize
        recovered = dequantize_cache(encoded, scale, mn, 'k', group_size, bits)

        # Check shape
        assert recovered.shape == original_shape

        # Check error is within reasonable bounds (2-bit is more lossy)
        error = torch.abs(k_cache - recovered).max().item()
        relative_error = error / (k_cache.abs().max().item() + 1e-6)
        print(f"K cache - group_size={group_size}, bits={bits}, max_error={error:.6f}, relative_error={relative_error:.6f}")
        # Allow higher error for 2-bit quantization (very aggressive compression)
        threshold = 0.35 if bits == 2 else 0.1
        assert relative_error < threshold

    def test_quantize_k_cache_invalid_t_dimension(self):
        """Test K cache quantization with T not divisible by groupSize."""
        k_cache = torch.randn(32, 255, 128, dtype=torch.float16)  # 255 not divisible by 32
        with pytest.raises(AssertionError):
            quantize_cache(k_cache, 'k', group_size=32, bits=4)


class TestQuantizeVCache:
    """Test V cache quantization and dequantization."""

    @pytest.mark.parametrize("group_size,bits", [
        (32, 4),
        (64, 4),
        (32, 2),
        (32, 8),
    ])
    def test_quantize_dequantize_v_cache(self, group_size, bits):
        """Test V cache quantization and dequantization."""
        nh, T = 32, 256
        D = group_size * 4  # Vary D based on group_size
        v_cache = torch.randn(nh, T, D, dtype=torch.float16)  # [nh, T, D]
        original_shape = v_cache.shape

        # Quantize
        encoded, scale, mn = quantize_cache(v_cache, 'v', group_size, bits)

        # Check shapes - both K and V should have the same encoded shape now!
        feat_per_int = 32 // bits
        num_groups = D // group_size
        assert encoded.shape == (nh, T // feat_per_int, D)  # [nh, T/feat, D]
        assert scale.shape == (nh, T, num_groups, 1)  # [nh, T, num_groups, 1]
        assert mn.shape == (nh, T, num_groups, 1)

        # Dequantize
        recovered = dequantize_cache(encoded, scale, mn, 'v', group_size, bits)

        # Check shape
        assert recovered.shape == original_shape

        # Check error is within reasonable bounds (2-bit is more lossy)
        error = torch.abs(v_cache - recovered).max().item()
        relative_error = error / (v_cache.abs().max().item() + 1e-6)
        print(f"V cache - group_size={group_size}, bits={bits}, max_error={error:.6f}, relative_error={relative_error:.6f}")
        # Allow higher error for 2-bit quantization (very aggressive compression)
        threshold = 0.35 if bits == 2 else 0.1
        assert relative_error < threshold

    def test_quantize_v_cache_invalid_d_dimension(self):
        """Test V cache quantization with D not divisible by groupSize."""
        v_cache = torch.randn(32, 256, 127, dtype=torch.float16)  # 127 not divisible by 32
        with pytest.raises(AssertionError):
            quantize_cache(v_cache, 'v', group_size=32, bits=4)


class TestUnifiedPacking:
    """Test that K and V have unified packing format."""

    def test_k_v_encoded_same_shape(self):
        """Test that K and V encoded tensors have the same shape."""
        k_cache = torch.randn(32, 256, 128, dtype=torch.float16)  # [nh, T, D]
        v_cache = torch.randn(32, 256, 128, dtype=torch.float16)  # [nh, T, D]

        for bits in [2, 4, 8]:
            k_encoded, k_scale, k_mn = quantize_cache(k_cache, 'k', 32, bits)
            v_encoded, v_scale, v_mn = quantize_cache(v_cache, 'v', 32, bits)

            # K and V encoded should have the SAME shape
            assert k_encoded.shape == v_encoded.shape, (
                f"K and V encoded shapes don't match for {bits}-bit: "
                f"K={k_encoded.shape}, V={v_encoded.shape}"
            )

            print(f"{bits}-bit: K and V both have encoded shape {k_encoded.shape}")

            # Check the shape is [nh, T/feat_per_int, D]
            feat_per_int = 32 // bits
            expected_shape = (32, 256 // feat_per_int, 128)
            assert k_encoded.shape == expected_shape
            assert v_encoded.shape == expected_shape


class TestQuantizationCompressionRatio:
    """Test compression ratio of quantization."""

    def test_compression_ratio_4bit(self):
        """Test 4-bit quantization achieves ~4x compression."""
        k_cache = torch.randn(32, 256, 128, dtype=torch.float16)
        v_cache = torch.randn(32, 256, 128, dtype=torch.float16)

        original_size = k_cache.numel() * 2 + v_cache.numel() * 2  # float16 = 2 bytes

        k_encoded, k_scale, k_mn = quantize_cache(k_cache, 'k', 32, 4)
        v_encoded, v_scale, v_mn = quantize_cache(v_cache, 'v', 32, 4)

        quantized_size = (
            k_encoded.numel() * 4 + k_scale.numel() * 2 + k_mn.numel() * 2 +
            v_encoded.numel() * 4 + v_scale.numel() * 2 + v_mn.numel() * 2
        )

        compression_ratio = original_size / quantized_size
        print(f"Original size: {original_size / 1024**2:.2f} MB")
        print(f"Quantized size: {quantized_size / 1024**2:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")

        # Account for scale/mn metadata overhead
        assert compression_ratio > 3.0  # Should be close to 4x, but overhead reduces it

    def test_compression_ratio_2bit(self):
        """Test 2-bit quantization achieves ~8x compression."""
        k_cache = torch.randn(32, 256, 128, dtype=torch.float16)

        original_size = k_cache.numel() * 2

        k_encoded, k_scale, k_mn = quantize_cache(k_cache, 'k', 32, 2)

        quantized_size = k_encoded.numel() * 4 + k_scale.numel() * 2 + k_mn.numel() * 2

        compression_ratio = original_size / quantized_size
        print(f"2-bit compression ratio: {compression_ratio:.2f}x")

        # Account for scale/mn metadata overhead (reduces from ideal 8x)
        assert compression_ratio > 5.0  # Should be close to 8x, but overhead reduces it


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_cache_type(self):
        """Test with invalid cache type."""
        cache = torch.randn(32, 256, 128, dtype=torch.float16)
        with pytest.raises(AssertionError, match="cache_type must be 'k' or 'v'"):
            quantize_cache(cache, 'x', 32, 4)

    def test_single_element(self):
        """Test with single element per group."""
        k_cache = torch.randn(1, 32, 32, dtype=torch.float16)  # [nh, T, D]
        encoded, scale, mn = quantize_cache(k_cache, 'k', 32, 4)
        recovered = dequantize_cache(encoded, scale, mn, 'k', 32, 4)

        # Single element should be reasonably reconstructed (allow for fp16 quantization error)
        # Note: even with single element, 4-bit quantization has limited precision
        max_abs_error = torch.abs(k_cache - recovered).max().item()
        relative_error = max_abs_error / (k_cache.abs().max().item() + 1e-6)
        assert relative_error < 0.15, f"Relative error {relative_error:.4f} exceeds threshold"

    def test_invalid_cache_type_dequantize(self):
        """Test dequantize with invalid cache type."""
        encoded = torch.randint(0, 16, (32, 32, 128), dtype=torch.int32)
        scale = torch.randn(32, 1, 1, 128, dtype=torch.float16)
        mn = torch.randn(32, 1, 1, 128, dtype=torch.float16)

        with pytest.raises(AssertionError, match="cache_type must be 'k' or 'v'"):
            dequantize_cache(encoded, scale, mn, 'x', 32, 4)


class TestRoundTrip:
    """Test quantization-dequantization roundtrip accuracy."""

    @pytest.mark.parametrize("cache_type,group_size,bits", [
        ('k', 32, 4),
        ('k', 64, 4),
        ('v', 32, 4),
        ('v', 64, 4),
        ('k', 32, 2),
        ('v', 32, 2),
    ])
    def test_roundtrip_accuracy(self, cache_type, group_size, bits):
        """Test that quantization-dequantization roundtrip preserves data reasonably."""
        cache = torch.randn(32, 256, 128, dtype=torch.float16)

        # Quantize
        encoded, scale, mn = quantize_cache(cache, cache_type, group_size, bits)

        # Dequantize
        recovered = dequantize_cache(encoded, scale, mn, cache_type, group_size, bits)

        # Check shape
        assert recovered.shape == cache.shape

        # Check accuracy
        mse = torch.mean((cache - recovered) ** 2)
        signal_variance = torch.var(cache)
        snr = 10 * torch.log10(signal_variance / (mse + 1e-10)).item()

        print(f"{cache_type.upper()} cache - group_size={group_size}, bits={bits}, SNR={snr:.2f} dB")

        # SNR thresholds: 4-bit should be > 15 dB, 2-bit has lower SNR (~8 dB)
        threshold = 6 if bits == 2 else 15
        assert snr > threshold, f"SNR {snr:.2f} dB is below threshold {threshold} dB"


class TestDtypeSupport:
    """Test that quantization works with different dtypes: FP16, FP32, BF16."""

    @pytest.mark.parametrize("dtype,bits", [
        (torch.float16, 4),
        (torch.float32, 4),
        (torch.bfloat16, 4),
    ])
    def test_quantize_dequantize_preserves_dtype(self, dtype, bits):
        """Test that quantization-dequantization preserves the original dtype."""
        cache = torch.randn(32, 256, 128, dtype=dtype)
        original_dtype = cache.dtype

        # Quantize
        encoded, scale, mn = quantize_cache(cache, 'k', 32, bits)

        # Check that scale and mn preserve the original dtype
        assert scale.dtype == original_dtype, f"scale dtype {scale.dtype} != original {original_dtype}"
        assert mn.dtype == original_dtype, f"mn dtype {mn.dtype} != original {original_dtype}"

        # Dequantize
        recovered = dequantize_cache(encoded, scale, mn, 'k', 32, bits)

        # Check that recovered tensor has the original dtype
        assert recovered.dtype == original_dtype, f"recovered dtype {recovered.dtype} != original {original_dtype}"

        # Check shape is preserved
        assert recovered.shape == cache.shape

        # Check accuracy (should be good for all dtypes)
        error = torch.abs(cache - recovered).max().item()
        relative_error = error / (cache.abs().max().item() + 1e-6)
        print(f"{dtype} quantization: max_error={error:.6f}, relative_error={relative_error:.6f}")
        assert relative_error < 0.15, f"Relative error {relative_error:.4f} too high for {dtype}"

    @pytest.mark.parametrize("cache_type", ['k', 'v'])
    def test_fp32_vs_fp16_accuracy(self, cache_type):
        """Compare FP32 vs FP16 quantization accuracy."""
        torch.manual_seed(42)  # For fair comparison

        # Use same data for both dtypes
        data_fp32 = torch.randn(32, 256, 128, dtype=torch.float32)
        data_fp16 = data_fp32.to(torch.float16)

        # Quantize both
        encoded_fp32, scale_fp32, mn_fp32 = quantize_cache(data_fp32, cache_type, 32, 4)
        encoded_fp16, scale_fp16, mn_fp16 = quantize_cache(data_fp16, cache_type, 32, 4)

        # Dequantize both
        recovered_fp32 = dequantize_cache(encoded_fp32, scale_fp32, mn_fp32, cache_type, 32, 4)
        recovered_fp16 = dequantize_cache(encoded_fp16, scale_fp16, mn_fp16, cache_type, 32, 4)

        # Check dtypes are preserved
        assert recovered_fp32.dtype == torch.float32
        assert recovered_fp16.dtype == torch.float16

        # FP32 should have slightly better accuracy
        mse_fp32 = torch.mean((data_fp32 - recovered_fp32) ** 2)
        mse_fp16 = torch.mean((data_fp16.to(torch.float32) - recovered_fp16.to(torch.float32)) ** 2)

        print(f"{cache_type.upper()} FP32 MSE: {mse_fp32:.8f}, FP16 MSE: {mse_fp16:.8f}")
        # Both should be reasonable
        assert mse_fp32 < 0.1
        assert mse_fp16 < 0.1


class TestNumericalStability:
    """Test numerical stability edge cases."""

    @pytest.mark.parametrize("cache_type,dtype", [
        ('k', torch.float16),
        ('k', torch.float32),
        ('v', torch.float16),
        ('v', torch.float32),
    ])
    def test_all_zeros(self, cache_type, dtype):
        """Test quantization handles all-zero tensors (e.g., padding tokens)."""
        # All zeros - common for padding tokens
        cache = torch.zeros(32, 256, 128, dtype=dtype)

        # Should not produce NaN or Inf
        encoded, scale, mn = quantize_cache(cache, cache_type, 32, 4)

        # Check no NaN or Inf in encoded
        assert not torch.isnan(encoded).any(), "Encoded contains NaN"
        assert not torch.isinf(encoded).any(), "Encoded contains Inf"

        # Scale should not be zero (due to epsilon)
        assert (scale > 0).all(), "Scale contains zero values"

        # Dequantize should work
        recovered = dequantize_cache(encoded, scale, mn, cache_type, 32, 4)

        # All zeros should quantize/dequantize to (near) zeros
        assert not torch.isnan(recovered).any(), "Recovered contains NaN"
        assert not torch.isinf(recovered).any(), "Recovered contains Inf"
        assert recovered.abs().max().item() < 1e-3, "All zeros should recover to near zeros"

    @pytest.mark.parametrize("cache_type,dtype", [
        ('k', torch.float16),
        ('k', torch.float32),
        ('v', torch.float16),
        ('v', torch.float32),
    ])
    def test_constant_values(self, cache_type, dtype):
        """Test quantization handles constant-value tensors."""
        # All same value (another edge case where mx == mn)
        cache = torch.ones(32, 256, 128, dtype=dtype) * 5.0

        # Should not produce NaN or Inf
        encoded, scale, mn = quantize_cache(cache, cache_type, 32, 4)

        # Check no NaN or Inf
        assert not torch.isnan(encoded).any(), "Encoded contains NaN"
        assert not torch.isinf(encoded).any(), "Encoded contains Inf"
        assert (scale > 0).all(), "Scale contains zero values"

        # Dequantize should work
        recovered = dequantize_cache(encoded, scale, mn, cache_type, 32, 4)

        # Constant values should be preserved reasonably
        assert not torch.isnan(recovered).any(), "Recovered contains NaN"
        assert not torch.isinf(recovered).any(), "Recovered contains Inf"

        # Error should be small for constant values
        error = torch.abs(cache - recovered).max().item()
        print(f"{cache_type.upper()} constant={cache[0,0,0].item():.2f}, error={error:.6f}")
        # With tiny epsilon scale, reconstruction should still be close
        assert error < 0.5, f"Constant value reconstruction error too large: {error}"

    @pytest.mark.parametrize("cache_type", ['k', 'v'])
    def test_mixed_zeros_and_values(self, cache_type):
        """Test quantization with mix of padding (zeros) and actual values."""
        cache = torch.randn(32, 256, 128, dtype=torch.float16)
        # Simulate padding: set first half of tokens to zero
        cache[:, :128, :] = 0

        # Should not produce NaN or Inf
        encoded, scale, mn = quantize_cache(cache, cache_type, 32, 4)
        recovered = dequantize_cache(encoded, scale, mn, cache_type, 32, 4)

        # Check no NaN or Inf
        assert not torch.isnan(recovered).any(), "Recovered contains NaN"
        assert not torch.isinf(recovered).any(), "Recovered contains Inf"

        # Non-zero part should have reasonable accuracy
        non_zero_mask = cache[:, 128:, :]
        if non_zero_mask.numel() > 0:
            non_zero_error = torch.abs(cache[:, 128:, :] - recovered[:, 128:, :]).max().item()
            assert non_zero_error < 1.0, f"Non-zero values error too large: {non_zero_error}"
