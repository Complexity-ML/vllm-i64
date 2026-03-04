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

    def _try_fused_quant(self, x: torch.Tensor):
        """
        Fused RMSNorm + INT8 quantization in 1 kernel pass.
        Returns (int8_tensor, scale_tensor) or None if unavailable.
        Eliminates the DRAM round-trip between norm and downstream INT8 matmul.
        """
        if not (x.is_cuda and x.dim() == 2):
            return None
        # Priority 1: CUDA I64_rmsnorm_quant_forward
        try:
            from vllm_i64.kernels.cuda import get_i64_cuda_ops
            cuda_ops = get_i64_cuda_ops()
            if cuda_ops is not None:
                return cuda_ops.rmsnorm_quant_forward(x, self.weight.data, self.eps)
        except (ImportError, AttributeError, Exception):
            pass
        # Priority 2: Triton fused RMSNorm+quant
        try:
            from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import triton_fused_rmsnorm_quant
            result = triton_fused_rmsnorm_quant(x, self.weight.data, self.eps)
            if result is not None:
                return result
        except ImportError:
            pass
        return None

    def _try_gpu_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        """Try CUDA → Triton fused RMSNorm on GPU. Returns None if unavailable."""
        if not (x.is_cuda and x.dim() == 2):
            return None
        # Priority 1: CUDA I64_rmsnorm
        try:
            from vllm_i64.kernels.cuda import get_i64_cuda_ops
            cuda_ops = get_i64_cuda_ops()
            if cuda_ops is not None:
                return cuda_ops.rmsnorm_forward(x, self.weight.data, self.eps)
        except (ImportError, Exception):
            pass
        # Priority 2: Triton fused RMSNorm
        try:
            from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import triton_fused_rmsnorm
            out = triton_fused_rmsnorm(x, self.weight.data, self.eps)
            if out is not None:
                return out
        except ImportError:
            pass
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'weight_q12'):
            return self._forward_integer(x)
        out = self._try_gpu_rmsnorm(x)
        if out is not None:
            return out
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

    def _forward_integer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integer RMSNorm: float rsqrt (irreducible) + INT32 weight multiply.

        On GPU: CUDA → Triton fused RMSNorm (faster than integer path).
        On CPU: Q7 normalized × Q12 weight → Q19, dequant to float.
        """
        out = self._try_gpu_rmsnorm(x)
        if out is not None:
            return out
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
