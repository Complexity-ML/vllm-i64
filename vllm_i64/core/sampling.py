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
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class TokenLogprob:
    """Log-probability info for a single generated token."""
    token_id: int
    logprob: float
    top_logprobs: Optional[Dict[int, float]] = None  # token_id -> logprob


@dataclass
class SampleOutput:
    """Result of sampling a batch — token IDs + optional logprobs."""
    token_ids: torch.Tensor        # (batch,) i64
    logprobs: Optional[List[Optional[TokenLogprob]]] = None


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

    # Logprobs
    logprobs: Optional[int] = None  # Number of top logprobs to return per token

    # Output constraints (logits processors)
    output_constraints: Optional[object] = None  # OutputConstraints from logits_processor


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


def apply_repetition_penalty_batch(
    logits: torch.Tensor,
    past_tokens_list: List[List[int]],
    penalty: float,
    vocab_size: int,
) -> torch.Tensor:
    """
    Apply repetition penalty to a batch of logits.

    For each request i, penalizes logits[i, token] for all tokens
    in past_tokens_list[i]. Positive logits are divided by penalty,
    negative logits are multiplied by penalty.

    Args:
        logits: (batch, vocab_size) — modified in-place
        past_tokens_list: per-request token history
        penalty: repetition penalty factor (1.0 = no penalty)
        vocab_size: filter out-of-range token IDs

    Returns:
        logits (modified in-place)
    """
    if penalty == 1.0:
        return logits

    device = logits.device
    for i in range(logits.shape[0]):
        past = past_tokens_list[i]
        if not past:
            continue
        past_tensor = torch.tensor(past, dtype=torch.long, device=device).unique()
        past_tensor = past_tensor[past_tensor < vocab_size]
        if past_tensor.numel() == 0:
            continue
        scores = logits[i, past_tensor]
        logits[i, past_tensor] = torch.where(
            scores > 0,
            scores / penalty,
            scores * penalty,
        )
    return logits


def sample_batch(
    logits: torch.Tensor,
    params: SamplingParams,
    past_tokens_list: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """
    Sample tokens for a batch.

    Args:
        logits: (batch, vocab_size)
        params: sampling parameters
        past_tokens_list: per-request token history (for repetition penalty)

    Returns:
        token_ids: (batch,) i64 tensor
    """
    # Apply repetition penalty before temperature/sampling
    if params.repetition_penalty != 1.0 and past_tokens_list is not None:
        logits = logits.float()
        apply_repetition_penalty_batch(
            logits, past_tokens_list, params.repetition_penalty, logits.shape[-1]
        )

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


def sample_batch_with_logprobs(
    logits: torch.Tensor,
    params: SamplingParams,
    past_tokens_list: Optional[List[List[int]]] = None,
) -> SampleOutput:
    """
    Sample tokens for a batch with optional logprob tracking.

    Computes log_softmax BEFORE top-k/top-p filtering so logprobs
    reflect the true model distribution (matches OpenAI API behavior).

    Args:
        logits: (batch, vocab_size)
        params: sampling parameters (with logprobs field)
        past_tokens_list: per-request token history

    Returns:
        SampleOutput with token_ids and optional logprobs
    """
    # Apply repetition penalty before anything
    if params.repetition_penalty != 1.0 and past_tokens_list is not None:
        logits = logits.float()
        apply_repetition_penalty_batch(
            logits, past_tokens_list, params.repetition_penalty, logits.shape[-1]
        )

    if params.temperature == 0.0:
        # Greedy — compute logprobs from raw logits
        token_ids = logits.argmax(dim=-1)
        if params.logprobs is not None:
            log_probs_all = torch.log_softmax(logits.float(), dim=-1)
            lp_list = _gather_logprobs(log_probs_all, token_ids, params.logprobs)
            return SampleOutput(token_ids=token_ids, logprobs=lp_list)
        return SampleOutput(token_ids=token_ids)

    logits = logits.float()

    if params.temperature != 1.0:
        logits = logits / params.temperature

    # Compute logprobs BEFORE top-k/top-p filtering (true model distribution)
    log_probs_all = None
    if params.logprobs is not None:
        log_probs_all = torch.log_softmax(logits, dim=-1)

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
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if log_probs_all is not None:
        lp_list = _gather_logprobs(log_probs_all, token_ids, params.logprobs)
        return SampleOutput(token_ids=token_ids, logprobs=lp_list)

    return SampleOutput(token_ids=token_ids)


def _gather_logprobs(
    log_probs: torch.Tensor,
    token_ids: torch.Tensor,
    num_top: int,
) -> List[TokenLogprob]:
    """
    Gather logprob info for sampled tokens.

    Args:
        log_probs: (batch, vocab_size) log-softmax output
        token_ids: (batch,) sampled token IDs
        num_top: number of top logprobs to include

    Returns:
        List of TokenLogprob, one per batch element
    """
    batch = token_ids.shape[0]
    results = []
    for i in range(batch):
        tid = int(token_ids[i].item())
        lp = float(log_probs[i, tid].item())

        top_lps = None
        if num_top > 0:
            k = min(num_top, log_probs.shape[-1])
            top_values, top_indices = log_probs[i].topk(k)
            top_lps = {
                int(top_indices[j].item()): float(top_values[j].item())
                for j in range(k)
            }

        results.append(TokenLogprob(token_id=tid, logprob=lp, top_logprobs=top_lps))
    return results


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
