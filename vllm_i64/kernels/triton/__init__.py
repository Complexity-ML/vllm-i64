"""
vllm-i64 :: Triton GPU Kernels

Fused operations that eliminate intermediate memory round-trips:
  - I64_fused_rmsnorm_quant: RMSNorm + INT8 quantize in one pass
  - I64_fused_silu_mul:      SiLU(gate) * up in one kernel
  - I64_fused_softmax:       Integer softmax (Q7 LUT-based)
  - I64_fused_rope:          Rotary embedding (fused cos/sin apply)
  - I64_fused_dequant_gemm:  INT8 dequant + GEMM fused

INL - 2025
"""

from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import (
    triton_fused_rmsnorm_quant,
    triton_fused_rmsnorm,
)
from vllm_i64.kernels.triton.I64_fused_silu_mul import (
    triton_fused_silu_mul,
)
from vllm_i64.kernels.triton.I64_fused_softmax import (
    triton_fused_softmax_integer,
)
from vllm_i64.kernels.triton.I64_fused_rope import (
    triton_fused_rope,
)
from vllm_i64.kernels.triton.I64_fused_dequant_gemm import (
    triton_dequant_gemm_int8,
)
from vllm_i64.kernels.triton.I64_int8_gemm import (
    triton_int8_gemm,
)
