"""
vllm-i64 :: I64_fused_silu_mul

Triton kernel: fused SiLU(gate) * up element-wise.

Standard PyTorch path (3 ops, 3 memory round-trips):
    1. sigmoid(gate)           → write to DRAM
    2. gate * sigmoid(gate)    → write SiLU output to DRAM
    3. silu_out * up           → write final to DRAM

Fused path (1 kernel):
    Load gate, up → SiLU(gate) * up → store result

Eliminates 2 intermediate tensors (each size N × inter).
For Llama-7B decode (inter=11008): saves ~44KB per token per layer.

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
    def _I64_silu_mul_kernel(
        gate_ptr,     # (N, I) float
        up_ptr,       # (N, I) float
        out_ptr,      # (N, I) float
        numel,
        BLOCK: tl.constexpr,
    ):
        """Vectorized SiLU(gate) * up — one program per element block."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # SiLU(gate) = gate * sigmoid(gate)
        silu = gate * tl.sigmoid(gate)
        result = silu * up

        tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


    @triton.jit
    def _I64_silu_mul_inplace_kernel(
        gate_ptr,     # (N, I) float — overwritten with result
        up_ptr,       # (N, I) float
        numel,
        BLOCK: tl.constexpr,
    ):
        """In-place variant: gate = SiLU(gate) * up. Saves output allocation."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        silu = gate * tl.sigmoid(gate)
        result = silu * up

        tl.store(gate_ptr + offsets, result.to(gate_ptr.dtype.element_ty), mask=mask)


def triton_fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    inplace: bool = False,
) -> Optional[torch.Tensor]:
    """
    Fused SiLU(gate) * up via Triton.

    Args:
        gate: (N, I) float — gate projection output
        up: (N, I) float — up projection output
        inplace: if True, overwrite gate tensor (saves allocation)

    Returns:
        result: (N, I) float = SiLU(gate) * up
        None if Triton not available
    """
    if not (_TRITON and gate.is_cuda):
        return None

    numel = gate.numel()
    BLOCK = 1024

    if inplace:
        grid = (triton.cdiv(numel, BLOCK),)
        _I64_silu_mul_inplace_kernel[grid](gate, up, numel, BLOCK=BLOCK)
        return gate
    else:
        out = torch.empty_like(gate)
        grid = (triton.cdiv(numel, BLOCK),)
        _I64_silu_mul_kernel[grid](gate, up, out, numel, BLOCK=BLOCK)
        return out
