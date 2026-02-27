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

    # Beam search
    num_beams: int = 1  # 1 = no beam search
    length_penalty: float = 1.0

    # Structured output
    json_mode: bool = False
    stop_token_ids: Optional[List[int]] = None


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


# =========================================================================
# Beam Search
# =========================================================================

@dataclass
class BeamHypothesis:
    """A single beam hypothesis — integer token sequence + float score."""
    token_ids: List[int]     # Generated token IDs (i64)
    score: float             # Log-probability sum (only float in beam search)
    is_finished: bool = False


class BeamSearcher:
    """
    Beam search for token-routed models.

    Maintains `num_beams` hypotheses and expands the best at each step.
    Compatible with deterministic i64 routing: routing is fixed per token,
    so beam search only affects which tokens are selected.

    Integer-first: token IDs and beam indices are integer.
    Only beam scores are float (log-probabilities).
    """

    def __init__(
        self,
        num_beams: int = 4,
        max_length: int = 256,
        length_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.eos_token_id = eos_token_id

        self.beams: List[BeamHypothesis] = []
        self.completed: List[BeamHypothesis] = []

    def init_beams(self, initial_token_ids: Optional[List[int]] = None):
        """Initialize beams with optional prefix."""
        prefix = initial_token_ids or []
        self.beams = [
            BeamHypothesis(token_ids=list(prefix), score=0.0)
            for _ in range(self.num_beams)
        ]
        self.completed = []

    def step(self, logits_per_beam: torch.Tensor) -> List[List[int]]:
        """
        One beam search step.

        Args:
            logits_per_beam: (num_beams, vocab_size) — logits from each beam's forward

        Returns:
            List of token ID sequences for each beam (for next forward pass)
        """
        vocab_size = logits_per_beam.shape[-1]
        log_probs = torch.log_softmax(logits_per_beam, dim=-1)

        # Score all possible next tokens for all beams
        all_scores = []
        all_tokens = []
        all_beam_idx = []

        for beam_idx, beam in enumerate(self.beams):
            if beam.is_finished:
                continue
            beam_log_probs = log_probs[beam_idx]
            scores = beam.score + beam_log_probs  # (vocab_size,)

            all_scores.append(scores)
            all_tokens.append(torch.arange(vocab_size, device=logits_per_beam.device))
            all_beam_idx.extend([beam_idx] * vocab_size)

        if not all_scores:
            return [beam.token_ids for beam in self.beams]

        all_scores = torch.cat(all_scores)
        all_tokens = torch.cat(all_tokens)

        # Select top-k candidates (2 * num_beams for diversity)
        k = min(2 * self.num_beams, all_scores.shape[0])
        top_scores, top_indices = all_scores.topk(k)

        new_beams = []
        for i in range(k):
            if len(new_beams) >= self.num_beams:
                break

            idx = top_indices[i].item()
            score = top_scores[i].item()
            token_id = all_tokens[idx].item()
            beam_idx = all_beam_idx[idx]

            new_ids = self.beams[beam_idx].token_ids + [token_id]

            # Length penalty
            length_factor = ((5.0 + len(new_ids)) / 6.0) ** self.length_penalty
            penalized_score = score / length_factor

            hypothesis = BeamHypothesis(
                token_ids=new_ids,
                score=penalized_score,
            )

            # Check for EOS
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                hypothesis.is_finished = True
                self.completed.append(hypothesis)
                continue

            # Check max length
            if len(new_ids) >= self.max_length:
                hypothesis.is_finished = True
                self.completed.append(hypothesis)
                continue

            new_beams.append(hypothesis)

        # Pad with remaining beams if needed
        while len(new_beams) < self.num_beams:
            new_beams.append(BeamHypothesis(token_ids=[], score=float("-inf"), is_finished=True))

        self.beams = new_beams
        return [beam.token_ids for beam in self.beams]

    @property
    def is_done(self) -> bool:
        """All beams finished."""
        return all(b.is_finished for b in self.beams)

    def get_best(self) -> BeamHypothesis:
        """Return the best completed hypothesis (or best active beam)."""
        candidates = self.completed + [b for b in self.beams if not b.is_finished]
        if not candidates:
            candidates = self.beams
        return max(candidates, key=lambda h: h.score)
