"""
vllm-i64 :: RMSNorm

Standard + integer RMS normalization. Model-agnostic.

Float path: standard x * rsqrt(mean(x^2) + eps) * weight.
Integer path (enabled by weight_q12 buffer):
  - Float rsqrt (1 scalar per token — irreducible, no integer analog)
  - INT32 weight multiply: normalized_q7 × weight_q12 → out_q19
  - Saves the float weight multiply on the final step

INL - 2025
"""

import torch
import torch.nn as nn

# Fixed-point scales for integer RMSNorm
_Q_NORM = 128      # Q7 for normalized values (~unit scale after RMSNorm)
_Q_WEIGHT = 4096   # Q12 for weights (~1.0, 12-bit resolution)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'weight_q12'):
            return self._forward_integer(x)
        # Try Triton fused RMSNorm on GPU
        if x.is_cuda and x.dim() == 2:
            try:
                from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import triton_fused_rmsnorm
                out = triton_fused_rmsnorm(x, self.weight.data, self.eps)
                if out is not None:
                    return out
            except ImportError:
                pass
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

    def _forward_integer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integer RMSNorm: float rsqrt (irreducible) + INT32 weight multiply.

        1. Compute variance + rsqrt in float (1 scalar op per token)
        2. Quantize normalized x to Q7 (×128 → INT32)
        3. Multiply by pre-quantized weight_q12 in INT32
        4. Dequant: ÷(128 × 4096) back to float
        """
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        xn = x.float() * norm
        xn_q7 = (xn * _Q_NORM).round().to(torch.int32)
        out_q19 = xn_q7 * self.weight_q12.to(torch.int32)
        return (out_q19.float() / (_Q_NORM * _Q_WEIGHT)).type_as(x)


def quantize_rmsnorm(module: "RMSNorm") -> None:
    """Quantize RMSNorm weight to Q12 INT16 for integer forward path."""
    w = module.weight.data.float()
    w_q12 = (w * _Q_WEIGHT).round().clamp(-32768, 32767).to(torch.int16)
    module.register_buffer('weight_q12', w_q12)
