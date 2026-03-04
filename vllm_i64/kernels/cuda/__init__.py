"""
vllm-i64 :: CUDA GPU Kernels (I64_* series)

JIT-compiled custom CUDA kernels for maximum GPU performance.
Falls back to Triton kernels, then to PyTorch ops.

Kernels (all with I64_ prefix):
  I64_rmsnorm.cu:   Fused RMSNorm + optional INT8 quantize
  I64_rope.cu:      Fused rotary embedding (float + Q14 integer)
  I64_quantize.cu:  INT8 activation/weight quantization
  I64_softmax.cu:   Integer softmax (Q7 LUT, constant memory)
  I64_gemm.cu:      Fused dequant+GEMM, fused gate+up+SiLU
  I64_binding.cu:   Unified pybind11 registration

INL - 2025
"""

from vllm_i64.kernels.cuda.I64_loader import (
    get_i64_cuda_ops,
    is_i64_cuda_available,
)
