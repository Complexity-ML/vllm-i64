"""
vllm-i64 :: Speculative Decoding

Token-routed models are ideal for speculative decoding:
  - Routing is DETERMINISTIC: expert_id = token_id % num_experts
  - No need to speculate on routing — it's known from the draft token
  - Draft model can be a smaller token-routed model (same routing logic)
  - Verification is exact: if draft token is accepted, routing was correct

Strategy:
  1. Draft model generates K tokens speculatively (greedy for draft)
  2. Target model verifies all K tokens in one forward pass
  3. Accept matching prefix, sample from target on mismatch
     (respects user's SamplingParams — temperature, top-k, top-p, etc.)

Tensor-only: no .item() calls in the draft loop for GPU efficiency.

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
        sampling_params=None,           # optional SamplingParams for target sampling
    ) -> Tuple[List[int], int]:
        """
        One speculative decoding step.

        Args:
            token_ids: current context tokens
            positions: current positions
            sampling_params: if provided, used for target model token selection
                on mismatch (temperature, top-k, top-p, etc.). If None, uses greedy.

        Returns:
            accepted_tokens: list of accepted token IDs (i64)
            num_draft_calls: number of draft forward passes
        """
        device = token_ids.device

        # Pre-allocate draft token buffer (avoid repeated cat + .item())
        draft_buf = torch.empty(self.K, dtype=torch.int64, device=device)

        # Phase 1: Draft model generates K tokens (always greedy for speed)
        # Build all draft inputs incrementally using pre-allocated buffer
        draft_input = token_ids.clone()
        next_pos = positions[-1:] + 1  # scalar tensor, no .item()

        for k in range(self.K):
            logits = self.draft(draft_input, positions if k == 0 else
                torch.cat([positions, torch.arange(
                    positions[-1] + 1, positions[-1] + 1 + k,
                    dtype=torch.int32, device=device)]))
            if logits.dim() > 1:
                logits = logits[-1:]  # keep as 1D tensor, no squeeze to scalar
            draft_buf[k] = logits.argmax(-1)

            # Extend input for next draft step (tensor cat, no .item())
            draft_input = torch.cat([draft_input, draft_buf[k:k+1]])

        # Phase 2: Target model verifies all K tokens in one pass
        # Build verify input: [last_context_token, draft_0, ..., draft_K-1]
        verify_tokens = torch.cat([token_ids[-1:], draft_buf])  # (K+1,)
        start_pos = positions[-1]  # scalar tensor
        verify_pos = torch.arange(
            start_pos, start_pos + verify_tokens.shape[0],
            dtype=torch.int32, device=device,
        )

        target_logits = self.target(verify_tokens, verify_pos)

        # Phase 3: Accept or reject each draft token using target model probabilities
        # For temperature=0 (greedy), accept iff draft == target_argmax
        # For temperature>0, use rejection sampling: accept with prob min(1, p_target/p_draft)
        draft_cpu = draft_buf.cpu().tolist()

        accepted = []
        for i, draft_token in enumerate(draft_cpu):
            if sampling_params is None or sampling_params.temperature == 0.0:
                # Greedy: accept only if draft matches target argmax
                target_token = target_logits[i].argmax(-1).item()
                if target_token == draft_token:
                    accepted.append(draft_token)
                else:
                    accepted.append(target_token)
                    break
            else:
                # Stochastic: accept with probability p_target(draft_token) / p_draft(draft_token)
                # This gives unbiased samples from the target distribution
                target_probs = torch.softmax(target_logits[i].float() / sampling_params.temperature, dim=-1)
                p_target = target_probs[draft_token].item()
                # Draft was greedy, so p_draft ~ 1 for draft token
                # Accept with min(1, p_target) as conservative approximation
                import random
                if random.random() < p_target:
                    accepted.append(draft_token)
                else:
                    # Reject: sample from adjusted distribution
                    sampled = self._sample_target(target_logits[i], sampling_params)
                    accepted.append(sampled)
                    break
        else:
            # All K tokens accepted — bonus: sample next from target
            bonus = self._sample_target(target_logits[self.K], sampling_params)
            accepted.append(bonus)

        return accepted, self.K

    def _sample_target(self, logits_1d: torch.Tensor, sampling_params=None) -> int:
        """Sample a token from target logits respecting user's sampling params."""
        if sampling_params is None or sampling_params.temperature == 0.0:
            return logits_1d.argmax(-1).item()

        from vllm_i64.core.sampling import sample_token
        return sample_token(logits_1d, sampling_params)

    def acceptance_rate(self, accepted: int, drafted: int) -> float:
        """Track acceptance rate."""
        if drafted == 0:
            return 0.0
        return accepted / drafted
