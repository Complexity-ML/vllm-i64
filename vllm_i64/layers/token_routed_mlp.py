"""
vllm-i64 :: Token-Routed MLP (Generic)

Pure i64 deterministic expert routing. Model-agnostic.
Supports Tensor Parallelism: experts sharded on intermediate dim.

Integer:
  - Routing: token_id & (num_experts - 1) → expert_id  (replicated, all ranks)
  - Scatter/gather: argsort indices

Float:
  - Expert SwiGLU compute only

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Optional

from vllm_i64.parallel.tensor_parallel import get_tp, all_reduce
from vllm_i64.kernels.fused_experts import (
    fused_token_routed_forward,
    fused_token_routed_forward_int8,
    fused_token_routed_forward_int4,
)


class TokenRoutedMLP(nn.Module):
    """
    Generic token-routed MLP with TP support.

    Routing (i64, replicated on all ranks):
        expert_id = token_id % num_experts

    Expert compute (float, sharded across TP ranks):
        gate_up: (E, hidden, 2 * inter_per_tp) — ColumnParallel
        down:    (E, inter_per_tp, hidden)      — RowParallel + all_reduce
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
    ):
        super().__init__()
        tp = get_tp()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.tp_size = tp.tp_size

        self.full_expert_inter = intermediate_size // num_experts
        self.expert_inter = self.full_expert_inter // tp.tp_size

        # Expert weights (sharded on intermediate dim)
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * self.expert_inter)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_inter, hidden_size)
        )

        # i64 routing table (replicated — cheap)
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def route(self, token_ids: Optional[torch.Tensor], num_tokens: int, device: torch.device) -> torch.Tensor:
        """Pure i64 routing. Override in subclasses."""
        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=device)
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        return self.token_to_expert[token_ids_clamped]

    def expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Dispatch + SwiGLU + all_reduce (fused)."""
        if hasattr(self, 'gate_up_int4'):
            output = fused_token_routed_forward_int4(
                x, self.gate_up_int4, self.gate_up_scale_int4, self.gate_up_zero,
                self.down_int4, self.down_scale_int4, self.down_zero,
                expert_ids, self.num_experts, self.expert_inter,
            )
        elif hasattr(self, 'gate_up_int8'):
            output = fused_token_routed_forward_int8(
                x, self.gate_up_int8, self.gate_up_scale,
                self.down_int8, self.down_scale,
                expert_ids, self.num_experts, self.expert_inter,
            )
        else:
            output = fused_token_routed_forward(
                x, self.gate_up_proj, self.down_proj,
                expert_ids, self.num_experts, self.expert_inter,
            )
        # All-reduce across TP ranks (RowParallel equivalent)
        return all_reduce(output)

    def forward(self, x, token_ids=None, **kwargs):
        expert_ids = self.route(token_ids, x.shape[0], x.device)
        return self.expert_forward(x, expert_ids)

    def load_full_weights(self, full_gate_up: torch.Tensor, full_down: torch.Tensor):
        """Load from unsharded checkpoint, take our TP slice."""
        from vllm_i64.parallel.tensor_parallel import shard_expert_weights
        gu_shard, dn_shard = shard_expert_weights(full_gate_up, full_down)
        self.gate_up_proj.data.copy_(gu_shard)
        self.down_proj.data.copy_(dn_shard)
