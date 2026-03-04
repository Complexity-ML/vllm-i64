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
        if hasattr(self, 'gate_up_int8'):
            return self._forward_int8_fused(x)
        if hasattr(self, 'gate_int8'):
            return self._forward_int8(x)
        if hasattr(self, 'gate_int4'):
            return self._forward_int4(x)

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def _forward_int8_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused gate+up: 1 quantization + 1 matmul, then down."""
        from vllm_i64.core.quantization import int8_fused_gate_up_native, int8_linear_native

        gate, up = int8_fused_gate_up_native(
            x, self.gate_up_int8, self.gate_up_scale, self.gate_up_inter,
        )
        inter = F.silu(gate) * up
        out = int8_linear_native(inter, self.down_int8, self.down_scale)
        return all_reduce(out)

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """Separate gate/up INT8 matmuls (no fused weights)."""
        from vllm_i64.core.quantization import int8_linear_native

        gate = int8_linear_native(x, self.gate_int8, self.gate_scale)
        up = int8_linear_native(x, self.up_int8, self.up_scale)
        inter = F.silu(gate) * up
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
