"""
vllm-i64 :: Token-Routed MLP (Generic)

Pure i64 deterministic expert routing. Model-agnostic.
No mu, no INL Dynamics — just token_id % num_experts → SwiGLU expert.

Any token-routed model can use this layer:
  - Complexity Deep (with mu override on top)
  - Any future model with deterministic token routing

Integer:
  - Routing: token_id & (num_experts - 1) → expert_id
  - Scatter/gather: argsort indices

Float:
  - Expert SwiGLU compute only

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenRoutedMLP(nn.Module):
    """
    Generic token-routed MLP.

    Routing:
        expert_id = token_id % num_experts  (i64, zero float)

    Expert compute (SwiGLU):
        silu(x @ gate_w) * (x @ up_w) → @ down_w

    Args:
        hidden_size: model hidden dimension
        intermediate_size: total intermediate size (split across experts)
        num_experts: number of experts
        vocab_size: vocabulary size (for token_to_expert buffer)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.expert_inter = intermediate_size // num_experts

        # Expert weights: gate+up fused, down separate
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * self.expert_inter)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_inter, hidden_size)
        )

        # i64 token → expert lookup table
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def route(self, token_ids: Optional[torch.Tensor], num_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Pure i64 routing. Override this in subclasses for custom routing.

        Returns: expert_ids (num_tokens,) long tensor
        """
        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=device)

        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        return self.token_to_expert[token_ids_clamped]

    def expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Dispatch tokens to experts and compute SwiGLU.
        Sorted by expert for coalesced memory access.
        """
        # Sort by expert (integer sort)
        sorted_indices = expert_ids.argsort(stable=True)
        sorted_x = x[sorted_indices]
        sorted_expert_ids = expert_ids[sorted_indices]

        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        # SwiGLU per expert (the ONLY float computation)
        output = torch.empty(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)

        for eid in range(self.num_experts):
            start = expert_offsets[eid].item()
            end = expert_offsets[eid + 1].item()
            if start == end:
                continue

            chunk = sorted_x[start:end]
            gu = chunk @ self.gate_up_proj[eid]
            gate = gu[..., :self.expert_inter]
            up = gu[..., self.expert_inter:]
            output[start:end] = F.silu(gate) * up @ self.down_proj[eid]

        # Unsort to original token order
        final = torch.empty_like(output)
        final[sorted_indices] = output
        return final

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (num_tokens, hidden_size)
            token_ids: (num_tokens,) i64 token IDs for routing
            **kwargs: ignored (for subclass compatibility)
        """
        expert_ids = self.route(token_ids, x.shape[0], x.device)
        return self.expert_forward(x, expert_ids)
