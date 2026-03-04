"""
vllm-i64 :: Dense MLP

Dense SwiGLU MLP — the degenerate case of i64 routing (modulo 1).

    expert_id = token_id % 1 = 0  (always expert 0, trivially)

Forward: y = down_proj(silu(gate_proj(x)) * up_proj(x))

No sort, no dispatch, no scatter-gather. Straight GEMM.
TP: gate_proj/up_proj are ColumnParallel, down_proj is RowParallel.

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_i64.parallel.tensor_parallel import (
    get_tp, ColumnParallelLinear, RowParallelLinear, all_reduce,
)


class DenseMLP(nn.Module):
    """
    Dense SwiGLU MLP — expert_id = token_id % 1 = 0 for all tokens.

    Separate gate/up projections match HuggingFace Llama naming
    for zero-remapping weight loading.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        tp = get_tp()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.inter_per_tp = intermediate_size // tp.tp_size

        # ColumnParallel: output dim sharded (hidden → inter_per_tp)
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        # RowParallel: input dim sharded (inter_per_tp → hidden) + all_reduce
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Dense SwiGLU forward.

        **kwargs absorbs token_ids, expert_ids, mu — dense ignores all routing args.
        """
        if hasattr(self, 'gate_fp8'):
            return self._forward_fp8(x)
        if hasattr(self, 'gate_up_int8'):
            return self._forward_int8_fused(x)
        if hasattr(self, 'gate_int8'):
            return self._forward_int8(x)
        if hasattr(self, 'gate_int4'):
            return self._forward_int4(x)

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Try Triton fused SiLU*up on GPU
        if gate.is_cuda:
            try:
                from vllm_i64.kernels.triton.I64_fused_silu_mul import triton_fused_silu_mul
                inter = triton_fused_silu_mul(gate, up)
                if inter is not None:
                    return self.down_proj(inter)
            except ImportError:
                pass
        return self.down_proj(F.silu(gate) * up)

    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """FP8 E4M3 forward — native tensor core ops on H100/Ada."""
        from vllm_i64.core.fp8 import fp8_linear

        gate = fp8_linear(x, self.gate_fp8, self.gate_fp8_scale)
        up = fp8_linear(x, self.up_fp8, self.up_fp8_scale)
        inter = F.silu(gate) * up
        out = fp8_linear(inter, self.down_fp8, self.down_fp8_scale)
        return all_reduce(out)

    def _forward_int8_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused gate+up: 1 quantization + 1 matmul, then down.
        Integer SiLU LUT + INT32 gate*up multiply."""
        from vllm_i64.core.quantization import int8_fused_gate_up_native, int8_linear_native
        from vllm_i64.layers.moe import silu_multiply_integer

        gate, up = int8_fused_gate_up_native(
            x, self.gate_up_int8, self.gate_up_scale, self.gate_up_inter,
        )
        inter = silu_multiply_integer(gate, up)
        out = int8_linear_native(inter, self.down_int8, self.down_scale)
        return all_reduce(out)

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """Separate gate/up INT8 matmuls (no fused weights).
        Integer SiLU LUT + INT32 gate*up multiply."""
        from vllm_i64.core.quantization import int8_linear_native
        from vllm_i64.layers.moe import silu_multiply_integer

        gate = int8_linear_native(x, self.gate_int8, self.gate_scale)
        up = int8_linear_native(x, self.up_int8, self.up_scale)
        inter = silu_multiply_integer(gate, up)
        out = int8_linear_native(inter, self.down_int8, self.down_scale)
        return all_reduce(out)

    def _forward_int4(self, x: torch.Tensor) -> torch.Tensor:
        """INT4 dequant + GEMM path."""
        from vllm_i64.core.quantization import int4_linear

        gate = int4_linear(x, self.gate_int4, self.gate_scale_int4, self.gate_zero)
        up = int4_linear(x, self.up_int4, self.up_scale_int4, self.up_zero)
        inter = F.silu(gate) * up
        out = int4_linear(inter, self.down_int4, self.down_scale_int4, self.down_zero)
        return all_reduce(out)
