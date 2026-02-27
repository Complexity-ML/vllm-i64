"""
vllm-i64 :: Sampling

Sampling strategies for token generation.
All outputs are integer (i64 token IDs).

Strategies:
  - greedy (argmax)
  - temperature scaling
  - top-k filtering
  - top-p (nucleus) filtering
  - repetition penalty

INL - 2025
"""

import torch
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling parameters — all thresholds are simple values."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_tokens: int = 256


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    past_tokens: Optional[List[int]] = None,
) -> int:
    """
    Sample a single token from logits.

    Args:
        logits: (vocab_size,) float tensor
        params: sampling parameters
        past_tokens: previously generated tokens (for repetition penalty)

    Returns:
        token_id: int (i64)
    """
    logits = logits.float()

    # Repetition penalty
    if params.repetition_penalty != 1.0 and past_tokens:
        token_set = torch.tensor(past_tokens, dtype=torch.long, device=logits.device)
        token_set = token_set.unique()
        penalty_logits = logits[token_set]
        # Penalize: reduce positive, amplify negative
        penalty_logits = torch.where(
            penalty_logits > 0,
            penalty_logits / params.repetition_penalty,
            penalty_logits * params.repetition_penalty,
        )
        logits[token_set] = penalty_logits

    # Temperature
    if params.temperature == 0.0:
        # Greedy
        return logits.argmax().item()

    if params.temperature != 1.0:
        logits = logits / params.temperature

    # Top-k filtering
    if params.top_k > 0 and params.top_k < logits.shape[0]:
        top_k_values, _ = logits.topk(params.top_k)
        threshold = top_k_values[-1]
        logits[logits < threshold] = float("-inf")

    # Top-p (nucleus) filtering
    if params.top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        # Remove tokens with cumulative probability above threshold
        mask = cumulative_probs - probs > params.top_p
        sorted_logits[mask] = float("-inf")

        # Unsort
        logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    # Sample
    probs = torch.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id


def sample_batch(
    logits: torch.Tensor,
    params: SamplingParams,
) -> torch.Tensor:
    """
    Sample tokens for a batch.

    Args:
        logits: (batch, vocab_size)
        params: sampling parameters

    Returns:
        token_ids: (batch,) i64 tensor
    """
    if params.temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits.float()

    if params.temperature != 1.0:
        logits = logits / params.temperature

    # Top-k
    if params.top_k > 0 and params.top_k < logits.shape[-1]:
        top_k_values, _ = logits.topk(params.top_k, dim=-1)
        threshold = top_k_values[..., -1:]
        logits[logits < threshold] = float("-inf")

    # Top-p
    if params.top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        mask = (cumulative - probs) > params.top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
