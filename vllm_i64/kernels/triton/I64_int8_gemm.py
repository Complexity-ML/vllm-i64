"""
vllm-i64 :: I64_int8_gemm

Triton kernel: native INT8 x INT8 -> INT32 GEMM with rescale.

Unlike I64_fused_dequant_gemm (which dequants to float then does float GEMM),
this kernel keeps both activations and weights in INT8 and uses integer
dot products via tl.dot, then rescales once at the end.

    y[m,n] = (sum_k x_int8[m,k] * w_int8[n,k]) * x_scale[m] * w_scale[n]

Works on all GPUs with Triton support (SM70+), no cuBLAS dependency.

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
    def _I64_int8_gemm_kernel(
        # INT8 activations + per-token scale
        x_ptr,              # (M, K) int8
        x_scale_ptr,        # (M,) float32 — per-token scale
        # INT8 weights + per-channel scale
        w_ptr,              # (K, N) int8 — already transposed (w_int8.t())
        w_scale_ptr,        # (N,) float32 — per-channel scale
        # Output
        out_ptr,            # (M, N) float
        # Dimensions
        M, N, K,
        # Strides
        stride_x_m, stride_x_k,
        stride_w_k, stride_w_n,
        stride_o_m, stride_o_n,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Native INT8 GEMM: accumulate in INT32, rescale to float at the end.

        y[m,n] = (sum_k x_int8[m,k] * w_int8[k,n]) * x_scale[m] * w_scale[n]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offsets < M
        n_mask = n_offsets < N

        # INT32 accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        # Loop over K dimension — all arithmetic stays in INT8/INT32
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            # Load x tile: (BLOCK_M, BLOCK_K) int8
            x_ptrs = x_ptr + m_offsets[:, None] * stride_x_m + k_offsets[None, :] * stride_x_k
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)

            # Load w tile: (BLOCK_K, BLOCK_N) int8 — already transposed
            w_ptrs = w_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
            w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)

            # INT8 dot → INT32 accumulate
            acc += tl.dot(x_tile, w_tile)

        # Rescale: float_out = int32_acc * x_scale[m] * w_scale[n]
        x_scale = tl.load(x_scale_ptr + m_offsets, mask=m_mask, other=0.0)
        w_scale = tl.load(w_scale_ptr + n_offsets, mask=n_mask, other=0.0)

        out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]

        # Store
        o_ptrs = out_ptr + m_offsets[:, None] * stride_o_m + n_offsets[None, :] * stride_o_n
        tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def triton_int8_gemm(
    x_int8: torch.Tensor,
    x_scale: torch.Tensor,
    weight_int8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Native INT8 GEMM via Triton.

    Args:
        x_int8: (M, K) int8 — quantized activations
        x_scale: (M,) float32 — per-token scale
        weight_int8_t: (K, N) int8 — transposed weight (weight_int8.t().contiguous())
        weight_scale: (N,) float32 — per-channel scale
        bias: optional (N,) float

    Returns:
        y: (M, N) float, or None if Triton not available
    """
    if not (_TRITON and x_int8.is_cuda):
        return None

    # tl.dot with INT8 inputs requires SM80+ (Ampere) for native INT8 tensor cores.
    # On SM70/SM75, Triton may silently cast to FP16, giving incorrect results.
    cap = torch.cuda.get_device_capability()
    if cap[0] < 8:
        return None

    M, K = x_int8.shape
    N = weight_int8_t.shape[1]

    out = torch.empty(M, N, device=x_int8.device, dtype=torch.float32)

    # Block sizes — must be powers of 2 for tl.dot
    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    # Minimum 16 for tl.dot on most GPUs
    BLOCK_M = max(16, BLOCK_M)
    BLOCK_N = max(16, BLOCK_N)
    BLOCK_K = max(16, BLOCK_K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _I64_int8_gemm_kernel[grid](
        x_int8,
        x_scale,
        weight_int8_t, weight_scale,
        out,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        weight_int8_t.stride(0), weight_int8_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    if bias is not None:
        out = out + bias

    return out
