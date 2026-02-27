"""
vllm-i64 :: Speculative Decoding

Token-routed models are ideal for speculative decoding:
  - Routing is DETERMINISTIC: expert_id = token_id % num_experts
  - No need to speculate on routing — it's known from the draft token
  - Draft model can be a smaller token-routed model (same routing logic)
  - Verification is exact: if draft token is accepted, routing was correct

Strategy:
  1. Draft model generates K tokens speculatively
  2. Target model verifies all K tokens in one forward pass
  3. Accept matching prefix, reject from first mismatch

INL - 2025
"""

import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    """Speculative decoding configuration."""
    num_speculative_tokens: int = 5   # draft K tokens ahead
    draft_model: Optional[object] = None  # smaller token-routed model


class SpeculativeDecoder:
    """
    Speculative decoding for token-routed models.

    Since routing is deterministic (token_id % num_experts),
    the draft model's routing decisions are always valid
    if the draft tokens match.
    """

    def __init__(
        self,
        target_model,
        draft_model,
        num_speculative: int = 5,
    ):
        self.target = target_model
        self.draft = draft_model
        self.K = num_speculative

    def generate_step(
        self,
        token_ids: torch.Tensor,       # current context (i64)
        positions: torch.Tensor,        # current positions (i32)
    ) -> Tuple[List[int], int]:
        """
        One speculative decoding step.

        Returns:
            accepted_tokens: list of accepted token IDs (i64)
            num_draft_calls: number of draft forward passes
        """
        device = token_ids.device
        draft_tokens = []

        # Phase 1: Draft model generates K tokens
        draft_input = token_ids.clone()
        draft_pos = positions.clone()

        for _ in range(self.K):
            logits = self.draft(draft_input, draft_pos)
            if logits.dim() > 1:
                logits = logits[-1]
            next_token = logits.argmax().item()
            draft_tokens.append(next_token)

            # Extend for next draft step
            draft_input = torch.cat([
                draft_input,
                torch.tensor([next_token], dtype=torch.int64, device=device)
            ])
            draft_pos = torch.cat([
                draft_pos,
                torch.tensor([draft_pos[-1].item() + 1], dtype=torch.int32, device=device)
            ])

        # Phase 2: Target model verifies all K tokens in one pass
        verify_tokens = torch.tensor(
            [token_ids[-1].item()] + draft_tokens,
            dtype=torch.int64, device=device,
        )
        verify_pos = torch.arange(
            positions[-1].item(),
            positions[-1].item() + len(verify_tokens),
            dtype=torch.int32, device=device,
        )

        target_logits = self.target(verify_tokens, verify_pos)

        # Phase 3: Accept matching prefix
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            target_token = target_logits[i].argmax().item()
            if target_token == draft_token:
                accepted.append(draft_token)
            else:
                # Mismatch: accept target's token instead, stop
                accepted.append(target_token)
                break
        else:
            # All K tokens accepted, also sample next from target
            bonus_token = target_logits[self.K].argmax().item()
            accepted.append(bonus_token)

        return accepted, self.K

    def acceptance_rate(self, accepted: int, drafted: int) -> float:
        """Track acceptance rate."""
        if drafted == 0:
            return 0.0
        return accepted / drafted
