"""
vllm-i64 :: I64_fused_dequant_gemm

Triton kernel: fused INT8 dequantization + GEMM.

Standard path (2 kernels):
    1. w_float = w_int8 * scale     → write full FP16 weight to DRAM
    2. y = x @ w_float              → read weight back, compute GEMM

Fused path (1 kernel):
    Load INT8 weight + scale → dequant in-register → accumulate GEMM → store output

Saves writing + reading the full FP16 weight tensor.
For a 4096×11008 projection: saves 86MB of memory bandwidth.

INL - 2025
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    _TRITON = True
except ImportError:
    _TRITON = False


if _TRITON:
    @triton.jit
    def _I64_dequant_gemm_int8_kernel(
        # Input activations
        x_ptr,              # (M, K) float
        # INT8 weight + scale
        w_int8_ptr,         # (N, K) int8
        w_scale_ptr,        # (N,) float — per-channel scale
        # Output
        out_ptr,            # (M, N) float
        # Dimensions
        M, N, K,
        # Strides
        stride_x_m, stride_x_k,
        stride_w_n, stride_w_k,
        stride_o_m, stride_o_n,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused INT8 dequant + GEMM: y[m,n] = sum_k(x[m,k] * w_int8[n,k] * scale[n])

        Loads INT8 weights (1 byte), dequants in-register, accumulates in FP32.
        One program handles a (BLOCK_M, BLOCK_N) output tile.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offsets < M
        n_mask = n_offsets < N

        # Load per-channel scale for this N tile
        scale = tl.load(w_scale_ptr + n_offsets, mask=n_mask, other=0.0)  # (BLOCK_N,)

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K dimension
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            # Load x tile: (BLOCK_M, BLOCK_K) float
            x_ptrs = x_ptr + m_offsets[:, None] * stride_x_m + k_offsets[None, :] * stride_x_k
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            # Load w tile: (BLOCK_N, BLOCK_K) int8
            w_ptrs = w_int8_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
            w_int8 = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)

            # Dequant in-register: w_float = w_int8 * scale[n]
            w_float = w_int8.to(tl.float32) * scale[:, None]

            # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
            # Note: need w transposed: (BLOCK_N, BLOCK_K)^T = (BLOCK_K, BLOCK_N)
            acc += tl.dot(x_tile, tl.trans(w_float))

        # Store output
        o_ptrs = out_ptr + m_offsets[:, None] * stride_o_m + n_offsets[None, :] * stride_o_n
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def triton_dequant_gemm_int8(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Fused INT8 dequant + GEMM via Triton: y = x @ W^T where W is INT8.

    Args:
        x: (M, K) float activations
        weight_int8: (N, K) int8 weights
        weight_scale: (N,) float per-channel scale
        bias: optional (N,) float

    Returns:
        y: (M, N) float, or None if Triton not available
    """
    if not (_TRITON and x.is_cuda):
        return None

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight_int8.shape[0]

    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _I64_dequant_gemm_int8_kernel[grid](
        x_2d,
        weight_int8, weight_scale,
        out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_int8.stride(0), weight_int8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    if bias is not None:
        out = out + bias

    return out.reshape(*orig_shape[:-1], N)
