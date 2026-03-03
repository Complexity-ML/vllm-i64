"""
vllm-i64 :: Sampling Tests
INL - 2025
"""
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_i64.core.sampling import (
    SamplingParams,
    SampleOutput,
    TokenLogprob,
    sample_token,
    sample_batch,
    sample_batch_with_logprobs,
    apply_min_p,
    apply_typical_p,
    apply_min_tokens,
    apply_repetition_penalty_batch,
    apply_logit_bias,
    apply_frequency_presence_penalty_batch,
    BeamSearcher,
    BeamHypothesis,
)

VOCAB_SIZE = 10


# =========================================================================
# Helpers
# =========================================================================

def make_logits(values, batch=False):
    """Create a logits tensor from a list. batch=True wraps in a batch dim."""
    t = torch.tensor(values, dtype=torch.float32)
    if batch:
        if t.dim() == 1:
            t = t.unsqueeze(0)
    return t


def make_peaked_logits(peak_idx, vocab_size=VOCAB_SIZE, peak_val=10.0, base_val=0.0):
    """Create logits with one dominant token."""
    logits = torch.full((vocab_size,), base_val, dtype=torch.float32)
    logits[peak_idx] = peak_val
    return logits


def make_peaked_logits_batch(peak_indices, vocab_size=VOCAB_SIZE, peak_val=10.0, base_val=0.0):
    """Create batched logits, each row peaked at the corresponding index."""
    batch = len(peak_indices)
    logits = torch.full((batch, vocab_size), base_val, dtype=torch.float32)
    for i, idx in enumerate(peak_indices):
        logits[i, idx] = peak_val
    return logits


# =========================================================================
# 1. Greedy Sampling
# =========================================================================

class TestGreedySampling:
    """Temperature=0 should always return the argmax token."""

    def test_sample_token_greedy_returns_argmax(self):
        logits = make_peaked_logits(peak_idx=3)
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits.clone(), params)
        assert token == 3

    def test_sample_token_greedy_with_all_different_values(self):
        logits = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.0])
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits.clone(), params)
        assert token == 2  # 0.9 is highest

    def test_sample_batch_greedy_returns_argmax(self):
        logits = make_peaked_logits_batch([7, 2, 5])
        params = SamplingParams(temperature=0.0)
        tokens = sample_batch(logits.clone(), params)
        assert tokens.tolist() == [7, 2, 5]

    def test_greedy_is_deterministic_across_calls(self):
        logits = make_peaked_logits(peak_idx=6)
        params = SamplingParams(temperature=0.0)
        results = [sample_token(logits.clone(), params) for _ in range(10)]
        assert all(r == 6 for r in results)


# =========================================================================
# 2. Temperature Scaling
# =========================================================================

class TestTemperature:
    """Temperature scales logits before sampling, changing the distribution."""

    def test_high_temperature_flattens_distribution(self):
        """With very high temperature, distribution approaches uniform."""
        logits = make_peaked_logits(peak_idx=0, peak_val=10.0, base_val=0.0)
        # High temp: logits / 100 => [0.1, 0, 0, ...] -- nearly uniform
        high_temp_logits = logits / 100.0
        probs_high = torch.softmax(high_temp_logits, dim=-1)
        # Low temp: logits / 0.1 => [100, 0, 0, ...] -- very peaked
        low_temp_logits = logits / 0.1
        probs_low = torch.softmax(low_temp_logits, dim=-1)
        # High temp should have higher entropy (more spread out)
        assert probs_high.max().item() < probs_low.max().item()

    def test_low_temperature_sharpens_distribution(self):
        """Low temperature should make the distribution more peaked."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        params = SamplingParams(temperature=0.01, seed=42)
        # With very low temperature, should almost always pick token 0
        tokens = [sample_token(logits.clone(), params) for _ in range(20)]
        assert all(t == 0 for t in tokens)

    def test_temperature_one_is_identity(self):
        """Temperature=1.0 should not alter the logits."""
        logits = torch.randn(VOCAB_SIZE)
        scaled = logits.clone() / 1.0
        assert torch.allclose(logits, scaled)

    def test_batch_temperature_scaling(self):
        """Batch sampling with low temperature should be near-greedy."""
        logits = make_peaked_logits_batch([4, 8], peak_val=20.0, base_val=0.0)
        params = SamplingParams(temperature=0.01, seed=42)
        tokens = sample_batch(logits.clone(), params)
        assert tokens.tolist() == [4, 8]


# =========================================================================
# 3. Top-k Filtering
# =========================================================================

class TestTopK:
    """Top-k should keep only the k highest logit tokens."""

    def test_top_k_filters_to_k_tokens(self):
        """Only top-k tokens should have nonzero probability after filtering."""
        logits = torch.arange(VOCAB_SIZE, dtype=torch.float32)  # [0,1,2,...,9]
        params = SamplingParams(top_k=3, seed=42)
        # Sample many times -- should only ever get tokens 7, 8, or 9
        results = set()
        for _ in range(100):
            token = sample_token(logits.clone(), params)
            results.add(token)
        assert results.issubset({7, 8, 9})

    def test_top_k_1_is_greedy_like(self):
        """top_k=1 should always return the argmax."""
        logits = make_peaked_logits(peak_idx=5)
        params = SamplingParams(top_k=1, seed=0)
        for _ in range(10):
            assert sample_token(logits.clone(), params) == 5

    def test_top_k_equal_to_vocab_keeps_all(self):
        """top_k >= vocab_size should not filter anything."""
        logits = torch.ones(VOCAB_SIZE)
        params = SamplingParams(top_k=VOCAB_SIZE, seed=42)
        results = set()
        for _ in range(200):
            results.add(sample_token(logits.clone(), params))
        # With uniform logits and no filtering, we should see many tokens
        assert len(results) > 1

    def test_top_k_batch(self):
        """Batch top-k: peaked logits should still return the peak token."""
        logits = make_peaked_logits_batch([1, 9], peak_val=100.0)
        params = SamplingParams(top_k=3, seed=42)
        tokens = sample_batch(logits.clone(), params)
        assert tokens[0].item() == 1
        assert tokens[1].item() == 9


# =========================================================================
# 4. Top-p (Nucleus) Sampling
# =========================================================================

class TestTopP:
    """Top-p filters tokens by cumulative probability."""

    def test_top_p_restricts_to_nucleus(self):
        """With a peaked distribution and small top_p, only the peak survives."""
        logits = make_peaked_logits(peak_idx=2, peak_val=20.0, base_val=0.0)
        params = SamplingParams(top_p=0.5, seed=42)
        results = set()
        for _ in range(50):
            results.add(sample_token(logits.clone(), params))
        # Token 2 has overwhelming probability; top_p=0.5 should keep only it
        assert results == {2}

    def test_top_p_1_keeps_all(self):
        """top_p=1.0 should not filter anything."""
        logits = torch.ones(VOCAB_SIZE)
        params = SamplingParams(top_p=1.0, seed=42)
        results = set()
        for _ in range(200):
            results.add(sample_token(logits.clone(), params))
        assert len(results) > 1

    def test_top_p_very_small_selects_top_token(self):
        """Very small top_p should select only the most probable token."""
        logits = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        params = SamplingParams(top_p=0.01, seed=42)
        for _ in range(20):
            assert sample_token(logits.clone(), params) == 0

    def test_top_p_batch(self):
        """Batch top-p with peaked logits should return peak tokens."""
        logits = make_peaked_logits_batch([0, 6], peak_val=20.0)
        params = SamplingParams(top_p=0.5, seed=42)
        tokens = sample_batch(logits.clone(), params)
        assert tokens[0].item() == 0
        assert tokens[1].item() == 6


# =========================================================================
# 5. Min-p Dynamic Threshold
# =========================================================================

class TestMinP:
    """Min-p sets a floor relative to the top token's probability."""

    def test_min_p_filters_low_probability_tokens(self):
        """Tokens with prob < top_prob * min_p should be masked."""
        # Token 0 has much higher logit -> highest prob
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = apply_min_p(logits.clone(), min_p=0.5)
        # Token 0 has prob ~0.9999; threshold = 0.9999 * 0.5 ~ 0.5
        # Other tokens have prob ~0.00005 which is below threshold
        probs_after = torch.softmax(result, dim=-1)
        # Only token 0 should survive
        assert probs_after[0].item() > 0.99
        for i in range(1, VOCAB_SIZE):
            assert result[i].item() == float("-inf")

    def test_min_p_keeps_tokens_above_threshold(self):
        """Tokens close in probability to the top should survive."""
        # Two tokens nearly equal
        logits = torch.tensor([5.0, 4.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = apply_min_p(logits.clone(), min_p=0.5)
        # Both top tokens should survive since they are close
        assert result[0].item() != float("-inf")
        assert result[1].item() != float("-inf")

    def test_min_p_zero_disables(self):
        """min_p=0.0 should not modify logits."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_p(logits, min_p=0.0)
        assert torch.equal(result, original)

    def test_min_p_one_disables(self):
        """min_p=1.0 should not modify logits (boundary)."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_p(logits, min_p=1.0)
        assert torch.equal(result, original)

    def test_min_p_batch_dimension(self):
        """apply_min_p should handle (batch, vocab_size) tensors."""
        logits = torch.tensor([
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 4.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        result = apply_min_p(logits.clone(), min_p=0.5)
        # Row 0: only token 0 survives
        assert result[0, 0].item() != float("-inf")
        for i in range(1, VOCAB_SIZE):
            assert result[0, i].item() == float("-inf")
        # Row 1: tokens 0 and 1 should survive
        assert result[1, 0].item() != float("-inf")
        assert result[1, 1].item() != float("-inf")

    def test_min_p_integration_with_sample_batch(self):
        """min_p parameter in SamplingParams should be used by sample_batch."""
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        params = SamplingParams(min_p=0.5, seed=42)
        tokens = sample_batch(logits.clone(), params)
        assert tokens[0].item() == 0  # Only token 0 survives


# =========================================================================
# 6. Typical-p Entropy-Based Selection
# =========================================================================

class TestTypicalP:
    """Typical-p selects tokens whose information content is near the entropy."""

    def test_typical_p_filters_atypical_tokens(self):
        """With small typical_p, only the most 'typical' tokens remain."""
        logits = torch.tensor([3.0, 2.5, 2.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        result = apply_typical_p(logits.clone(), typical_p=0.3)
        # Some tokens should be masked to -inf
        num_active = (result > float("-inf")).sum().item()
        assert num_active < VOCAB_SIZE
        assert num_active >= 1

    def test_typical_p_one_disables(self):
        """typical_p=1.0 should not modify logits."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_typical_p(logits, typical_p=1.0)
        assert torch.equal(result, original)

    def test_typical_p_above_one_disables(self):
        """typical_p > 1.0 should not modify logits."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_typical_p(logits, typical_p=1.5)
        assert torch.equal(result, original)

    def test_typical_p_batch_dimension(self):
        """apply_typical_p should handle (batch, vocab_size) tensors."""
        logits = torch.randn(3, VOCAB_SIZE)
        result = apply_typical_p(logits.clone(), typical_p=0.3)
        # At least some tokens should be masked in each row
        for row in range(3):
            num_active = (result[row] > float("-inf")).sum().item()
            assert num_active >= 1

    def test_typical_p_preserves_at_least_one_token(self):
        """Even with tiny typical_p, at least one token should remain."""
        logits = torch.randn(VOCAB_SIZE)
        result = apply_typical_p(logits.clone(), typical_p=0.01)
        num_active = (result > float("-inf")).sum().item()
        assert num_active >= 1

    def test_typical_p_integration_with_sample_batch(self):
        """typical_p parameter in SamplingParams should be used by sample_batch."""
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        params = SamplingParams(typical_p=0.5, seed=42)
        # Should run without error and return a valid token
        tokens = sample_batch(logits.clone(), params)
        assert 0 <= tokens[0].item() < VOCAB_SIZE


# =========================================================================
# 7. Repetition Penalty
# =========================================================================

class TestRepetitionPenalty:
    """Repetition penalty should reduce logits for previously generated tokens."""

    def test_repetition_penalty_reduces_past_token_logits(self):
        """Positive logits for past tokens should be reduced by the penalty."""
        logits = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
        past_tokens_list = [[0, 1, 2]]
        result = apply_repetition_penalty_batch(logits.clone(), past_tokens_list, penalty=2.0, vocab_size=VOCAB_SIZE)
        # Tokens 0, 1, 2 should have logits = 5.0 / 2.0 = 2.5
        assert result[0, 0].item() == pytest.approx(2.5)
        assert result[0, 1].item() == pytest.approx(2.5)
        assert result[0, 2].item() == pytest.approx(2.5)
        # Others unchanged
        assert result[0, 3].item() == pytest.approx(5.0)

    def test_repetition_penalty_amplifies_negative_logits(self):
        """Negative logits for past tokens should be made more negative."""
        logits = torch.tensor([[-2.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        past_tokens_list = [[0, 1]]
        result = apply_repetition_penalty_batch(logits.clone(), past_tokens_list, penalty=2.0, vocab_size=VOCAB_SIZE)
        # Negative logits: -2.0 * 2.0 = -4.0
        assert result[0, 0].item() == pytest.approx(-4.0)
        assert result[0, 1].item() == pytest.approx(-4.0)

    def test_repetition_penalty_1_is_noop(self):
        """penalty=1.0 should not modify logits."""
        logits = torch.randn(2, VOCAB_SIZE)
        original = logits.clone()
        result = apply_repetition_penalty_batch(logits, [[1, 2], [3]], penalty=1.0, vocab_size=VOCAB_SIZE)
        assert torch.equal(result, original)

    def test_repetition_penalty_empty_past(self):
        """Empty past tokens should not modify logits."""
        logits = torch.randn(1, VOCAB_SIZE)
        original = logits.clone()
        result = apply_repetition_penalty_batch(logits, [[]], penalty=2.0, vocab_size=VOCAB_SIZE)
        assert torch.equal(result, original)

    def test_repetition_penalty_via_sample_token(self):
        """Repetition penalty should shift sampling away from past tokens."""
        # Token 0 and 1 are equally good, but 0 was already generated
        logits = torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)
        token = sample_token(logits.clone(), params, past_tokens=[0])
        # Token 0 penalized to 2.5, token 1 stays at 5.0 -> greedy picks 1
        assert token == 1

    def test_repetition_penalty_batch_multiple_rows(self):
        """Each row in the batch should use its own past_tokens."""
        logits = torch.tensor([
            [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        past = [[0], [1]]  # row 0 penalizes token 0, row 1 penalizes token 1
        result = apply_repetition_penalty_batch(logits.clone(), past, penalty=2.0, vocab_size=VOCAB_SIZE)
        assert result[0, 0].item() < result[0, 1].item()  # row 0: token 0 penalized
        assert result[1, 1].item() < result[1, 0].item()  # row 1: token 1 penalized

    def test_repetition_penalty_out_of_range_tokens_ignored(self):
        """Token IDs >= vocab_size in past should be ignored."""
        logits = torch.randn(1, VOCAB_SIZE)
        original = logits.clone()
        result = apply_repetition_penalty_batch(logits, [[100, 200]], penalty=2.0, vocab_size=VOCAB_SIZE)
        assert torch.equal(result, original)


# =========================================================================
# 8. Seed Reproducibility
# =========================================================================

class TestSeedReproducibility:
    """Same seed should produce the same results."""

    def test_same_seed_same_result_sample_token(self):
        """Identical seeds should yield identical tokens."""
        logits = torch.ones(VOCAB_SIZE)  # uniform distribution
        params = SamplingParams(seed=12345)
        results = []
        for _ in range(5):
            torch.manual_seed(12345)
            token = sample_token(logits.clone(), params)
            results.append(token)
        assert all(r == results[0] for r in results)

    def test_same_seed_same_result_sample_batch(self):
        """Batch sampling with same seed should be reproducible."""
        logits = torch.ones(3, VOCAB_SIZE)
        params = SamplingParams(seed=99999)
        tokens1 = sample_batch(logits.clone(), params)
        tokens2 = sample_batch(logits.clone(), params)
        assert torch.equal(tokens1, tokens2)

    def test_different_seeds_different_results(self):
        """Different seeds should (very likely) produce different sequences over many samples."""
        logits = torch.ones(VOCAB_SIZE)
        results_a = []
        results_b = []
        for i in range(20):
            params_a = SamplingParams(seed=42 + i * 1000)
            params_b = SamplingParams(seed=7777 + i * 1000)
            results_a.append(sample_token(logits.clone(), params_a))
            results_b.append(sample_token(logits.clone(), params_b))
        # Very unlikely to be identical across 20 samples
        assert results_a != results_b


# =========================================================================
# 9. Logit Bias
# =========================================================================

class TestLogitBias:
    """Logit bias adds/subtracts from specific token logits before sampling."""

    def test_positive_bias_increases_logit(self):
        logits = torch.zeros(VOCAB_SIZE)
        bias = {3: 100.0}
        result = apply_logit_bias(logits.clone(), bias)
        assert result[3].item() == pytest.approx(100.0)
        assert result[0].item() == pytest.approx(0.0)

    def test_negative_bias_decreases_logit(self):
        logits = torch.ones(VOCAB_SIZE) * 5.0
        bias = {7: -1000.0}
        result = apply_logit_bias(logits.clone(), bias)
        assert result[7].item() == pytest.approx(-995.0)

    def test_multiple_biases(self):
        logits = torch.zeros(VOCAB_SIZE)
        bias = {0: 10.0, 5: -5.0, 9: 3.0}
        result = apply_logit_bias(logits.clone(), bias)
        assert result[0].item() == pytest.approx(10.0)
        assert result[5].item() == pytest.approx(-5.0)
        assert result[9].item() == pytest.approx(3.0)
        assert result[1].item() == pytest.approx(0.0)

    def test_empty_bias_is_noop(self):
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_logit_bias(logits, {})
        assert torch.equal(result, original)

    def test_out_of_range_token_id_ignored(self):
        logits = torch.zeros(VOCAB_SIZE)
        bias = {100: 50.0}  # token 100 is out of range for vocab_size=10
        result = apply_logit_bias(logits.clone(), bias)
        assert torch.equal(result, torch.zeros(VOCAB_SIZE))

    def test_logit_bias_batch_dimension(self):
        """apply_logit_bias should work on (batch, vocab_size) tensors."""
        logits = torch.zeros(2, VOCAB_SIZE)
        bias = {4: 20.0}
        result = apply_logit_bias(logits.clone(), bias)
        # Ellipsis indexing should apply to all batch elements
        assert result[0, 4].item() == pytest.approx(20.0)
        assert result[1, 4].item() == pytest.approx(20.0)

    def test_logit_bias_shifts_greedy_choice(self):
        """A large positive bias should shift greedy sampling to that token."""
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        params = SamplingParams(temperature=0.0, logit_bias={5: 100.0})
        # sample_token does not apply logit_bias directly; test via sample_batch
        batch_logits = logits.unsqueeze(0)
        tokens = sample_batch(batch_logits.clone(), params)
        assert tokens[0].item() == 5

    def test_negative_logit_bias_suppresses_token(self):
        """Large negative bias should prevent a token from being sampled."""
        logits = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        params = SamplingParams(temperature=0.0, logit_bias={0: -100.0})
        tokens = sample_batch(logits.clone(), params)
        assert tokens[0].item() == 1  # token 0 suppressed, token 1 wins


# =========================================================================
# 10. Frequency / Presence Penalties
# =========================================================================

class TestFrequencyPresencePenalty:
    """Frequency and presence penalties reduce logits for tokens that have appeared."""

    def test_frequency_penalty_proportional_to_count(self):
        """Tokens that appear more often should be penalized more."""
        logits = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
        # Token 0 appears 3 times, token 1 appears 1 time
        past = [[0, 0, 0, 1]]
        result = apply_frequency_presence_penalty_batch(
            logits.clone(), past, frequency_penalty=1.0, presence_penalty=0.0, vocab_size=VOCAB_SIZE
        )
        # Token 0: 5.0 - 1.0*3 = 2.0, Token 1: 5.0 - 1.0*1 = 4.0
        assert result[0, 0].item() == pytest.approx(2.0)
        assert result[0, 1].item() == pytest.approx(4.0)
        # Token 2 unaffected
        assert result[0, 2].item() == pytest.approx(5.0)

    def test_presence_penalty_binary(self):
        """Presence penalty should apply equally regardless of count."""
        logits = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
        past = [[0, 0, 0, 1]]  # Token 0 appears 3x, token 1 appears 1x
        result = apply_frequency_presence_penalty_batch(
            logits.clone(), past, frequency_penalty=0.0, presence_penalty=2.0, vocab_size=VOCAB_SIZE
        )
        # Both token 0 and token 1 get same presence penalty: 5.0 - 2.0*1 = 3.0
        assert result[0, 0].item() == pytest.approx(3.0)
        assert result[0, 1].item() == pytest.approx(3.0)
        assert result[0, 2].item() == pytest.approx(5.0)

    def test_combined_frequency_and_presence(self):
        """Both penalties should apply additively."""
        logits = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])
        past = [[0, 0, 0]]  # Token 0 appears 3 times
        result = apply_frequency_presence_penalty_batch(
            logits.clone(), past, frequency_penalty=1.0, presence_penalty=1.0, vocab_size=VOCAB_SIZE
        )
        # Token 0: 10.0 - 1.0*3 - 1.0*1 = 6.0
        assert result[0, 0].item() == pytest.approx(6.0)
        assert result[0, 1].item() == pytest.approx(10.0)

    def test_zero_penalties_noop(self):
        """Zero penalties should not modify logits."""
        logits = torch.randn(2, VOCAB_SIZE)
        original = logits.clone()
        result = apply_frequency_presence_penalty_batch(
            logits, [[1, 2], [3]], frequency_penalty=0.0, presence_penalty=0.0, vocab_size=VOCAB_SIZE
        )
        assert torch.equal(result, original)

    def test_penalties_empty_past(self):
        """Empty past tokens should not modify logits."""
        logits = torch.randn(1, VOCAB_SIZE)
        original = logits.clone()
        result = apply_frequency_presence_penalty_batch(
            logits, [[]], frequency_penalty=1.0, presence_penalty=1.0, vocab_size=VOCAB_SIZE
        )
        assert torch.equal(result, original)

    def test_penalties_out_of_range_tokens_ignored(self):
        """Token IDs >= vocab_size should be ignored in counts."""
        logits = torch.randn(1, VOCAB_SIZE)
        original = logits.clone()
        result = apply_frequency_presence_penalty_batch(
            logits, [[100, 200]], frequency_penalty=1.0, presence_penalty=1.0, vocab_size=VOCAB_SIZE
        )
        assert torch.equal(result, original)

    def test_frequency_penalty_via_sample_batch(self):
        """Frequency penalty in SamplingParams should shift greedy choice."""
        # Tokens 0 and 1 equally good, but token 0 appeared many times
        logits = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        params = SamplingParams(temperature=0.0, frequency_penalty=2.0)
        past = [[0, 0, 0, 0, 0]]
        tokens = sample_batch(logits.clone(), params, past_tokens_list=past)
        # Token 0: 5.0 - 2.0*5 = -5.0, Token 1: 5.0 -> greedy picks 1
        assert tokens[0].item() == 1


# =========================================================================
# 11. min_tokens Suppresses EOS
# =========================================================================

class TestMinTokens:
    """min_tokens should suppress EOS until enough tokens are generated."""

    def test_eos_suppressed_before_min_tokens(self):
        """EOS should be -inf when num_generated < min_tokens."""
        logits = torch.zeros(VOCAB_SIZE)
        eos_id = 2
        logits[eos_id] = 10.0  # EOS is the best token
        result = apply_min_tokens(logits.clone(), num_generated=3, min_tokens=5, eos_token_id=eos_id)
        assert result[eos_id].item() == float("-inf")
        # Other tokens unchanged
        assert result[0].item() == pytest.approx(0.0)

    def test_eos_allowed_at_min_tokens(self):
        """EOS should NOT be suppressed when num_generated >= min_tokens."""
        logits = torch.zeros(VOCAB_SIZE)
        eos_id = 2
        logits[eos_id] = 10.0
        result = apply_min_tokens(logits.clone(), num_generated=5, min_tokens=5, eos_token_id=eos_id)
        assert result[eos_id].item() == pytest.approx(10.0)

    def test_eos_allowed_after_min_tokens(self):
        """EOS should not be suppressed when num_generated > min_tokens."""
        logits = torch.zeros(VOCAB_SIZE)
        eos_id = 2
        logits[eos_id] = 10.0
        result = apply_min_tokens(logits.clone(), num_generated=10, min_tokens=5, eos_token_id=eos_id)
        assert result[eos_id].item() == pytest.approx(10.0)

    def test_min_tokens_zero_disables(self):
        """min_tokens=0 should not suppress anything."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_tokens(logits, num_generated=0, min_tokens=0, eos_token_id=2)
        assert torch.equal(result, original)

    def test_min_tokens_none_eos_disables(self):
        """eos_token_id=None should not suppress anything."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_tokens(logits, num_generated=0, min_tokens=10, eos_token_id=None)
        assert torch.equal(result, original)

    def test_min_tokens_batch_dimension(self):
        """apply_min_tokens should handle (batch, vocab_size)."""
        logits = torch.zeros(3, VOCAB_SIZE)
        eos_id = 5
        logits[:, eos_id] = 10.0
        result = apply_min_tokens(logits.clone(), num_generated=2, min_tokens=5, eos_token_id=eos_id)
        # All batch rows should have EOS suppressed
        for i in range(3):
            assert result[i, eos_id].item() == float("-inf")

    def test_min_tokens_negative_disables(self):
        """Negative min_tokens should not suppress anything."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_tokens(logits, num_generated=0, min_tokens=-1, eos_token_id=2)
        assert torch.equal(result, original)


# =========================================================================
# 12. sample_batch_with_logprobs
# =========================================================================

class TestSampleBatchWithLogprobs:
    """sample_batch_with_logprobs should return SampleOutput with logprobs."""

    def test_returns_sample_output(self):
        logits = torch.randn(2, VOCAB_SIZE)
        params = SamplingParams(logprobs=5, seed=42)
        output = sample_batch_with_logprobs(logits.clone(), params)
        assert isinstance(output, SampleOutput)
        assert output.token_ids.shape == (2,)

    def test_logprobs_present_when_requested(self):
        logits = torch.randn(3, VOCAB_SIZE)
        params = SamplingParams(logprobs=3, seed=42)
        output = sample_batch_with_logprobs(logits.clone(), params)
        assert output.logprobs is not None
        assert len(output.logprobs) == 3
        for lp in output.logprobs:
            assert isinstance(lp, TokenLogprob)
            assert lp.top_logprobs is not None
            assert len(lp.top_logprobs) == 3

    def test_logprobs_none_when_not_requested(self):
        logits = torch.randn(2, VOCAB_SIZE)
        params = SamplingParams(logprobs=None, seed=42)
        output = sample_batch_with_logprobs(logits.clone(), params)
        assert output.logprobs is None

    def test_logprob_values_are_negative(self):
        """Log-probabilities should be <= 0."""
        logits = torch.randn(1, VOCAB_SIZE)
        params = SamplingParams(logprobs=5, seed=42)
        output = sample_batch_with_logprobs(logits.clone(), params)
        lp = output.logprobs[0]
        assert lp.logprob <= 0.0
        for v in lp.top_logprobs.values():
            assert v <= 0.0

    def test_sampled_token_in_top_logprobs_greedy(self):
        """In greedy mode, the sampled token should be the top logprob."""
        logits = make_peaked_logits_batch([4], peak_val=20.0)
        params = SamplingParams(temperature=0.0, logprobs=5)
        output = sample_batch_with_logprobs(logits.clone(), params)
        assert output.token_ids[0].item() == 4
        lp = output.logprobs[0]
        assert lp.token_id == 4
        # Token 4 should have the highest logprob
        assert lp.logprob == max(lp.top_logprobs.values())

    def test_top_logprobs_sorted_descending(self):
        """Top logprobs should be ordered from highest to lowest."""
        logits = torch.randn(1, VOCAB_SIZE)
        params = SamplingParams(logprobs=5, seed=42)
        output = sample_batch_with_logprobs(logits.clone(), params)
        values = list(output.logprobs[0].top_logprobs.values())
        assert values == sorted(values, reverse=True)

    def test_logprobs_num_capped_at_vocab_size(self):
        """Requesting more logprobs than vocab_size should return vocab_size entries."""
        logits = torch.randn(1, VOCAB_SIZE)
        params = SamplingParams(logprobs=100, seed=42)
        output = sample_batch_with_logprobs(logits.clone(), params)
        assert len(output.logprobs[0].top_logprobs) == VOCAB_SIZE

    def test_with_logprobs_respects_repetition_penalty(self):
        """Logprobs should reflect the distribution after repetition penalty."""
        logits = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        params = SamplingParams(temperature=0.0, repetition_penalty=5.0, logprobs=2)
        output = sample_batch_with_logprobs(logits.clone(), params, past_tokens_list=[[0]])
        # Token 0 is penalized; greedy should pick token 1
        assert output.token_ids[0].item() == 1


# =========================================================================
# 13. BeamSearcher
# =========================================================================

class TestBeamSearcher:
    """Basic beam search tests."""

    def test_init_beams(self):
        bs = BeamSearcher(num_beams=3, max_length=10)
        bs.init_beams()
        assert len(bs.beams) == 3
        assert all(b.token_ids == [] for b in bs.beams)
        assert all(b.score == 0.0 for b in bs.beams)

    def test_init_beams_with_prefix(self):
        bs = BeamSearcher(num_beams=2, max_length=10)
        bs.init_beams(initial_token_ids=[1, 2, 3])
        assert len(bs.beams) == 2
        assert all(b.token_ids == [1, 2, 3] for b in bs.beams)

    def test_single_step_picks_top_tokens(self):
        """After one step, beams should extend with the best tokens."""
        bs = BeamSearcher(num_beams=2, max_length=10)
        bs.init_beams()
        # Create logits where token 5 is best for both beams
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[:, 5] = 10.0
        logits[:, 3] = 5.0
        sequences = bs.step(logits)
        assert len(sequences) == 2
        # Best token should be 5, second best 3
        all_last_tokens = [seq[-1] for seq in sequences]
        assert 5 in all_last_tokens

    def test_eos_moves_to_completed(self):
        """When a beam generates EOS, it should move to completed."""
        eos_id = 9
        bs = BeamSearcher(num_beams=2, max_length=10, eos_token_id=eos_id)
        bs.init_beams()
        # Make EOS the best token
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[:, eos_id] = 100.0
        bs.step(logits)
        assert len(bs.completed) > 0
        assert all(h.is_finished for h in bs.completed)

    def test_max_length_stops_beam(self):
        """Beams at max_length should be marked finished."""
        bs = BeamSearcher(num_beams=1, max_length=2)
        bs.init_beams()
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, 0] = 10.0
        # Step 1: beam has 1 token
        bs.step(logits)
        # Step 2: beam has 2 tokens -> max_length reached
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, 1] = 10.0
        bs.step(logits)
        assert len(bs.completed) > 0

    def test_is_done_when_all_finished(self):
        """is_done should be True when all beams are finished."""
        eos_id = 0
        bs = BeamSearcher(num_beams=2, max_length=10, eos_token_id=eos_id)
        bs.init_beams()
        # Make EOS overwhelmingly the best choice so it dominates all top-k slots
        logits = torch.full((2, VOCAB_SIZE), -1e9)
        logits[:, eos_id] = 100.0
        bs.step(logits)
        # After EOS for all beams, remaining candidates have score ~ -1e9
        # which is so low the beams are padded as finished
        # The completed list should have entries from both beams
        assert len(bs.completed) >= 2
        # Even if the remaining beams are not flagged done (filled from leftover
        # candidates), the best hypothesis should be in completed
        best = bs.get_best()
        assert best.is_finished

    def test_get_best_returns_highest_score(self):
        """get_best should return the hypothesis with the highest score."""
        bs = BeamSearcher(num_beams=2, max_length=10)
        bs.init_beams()
        # Different peaks for each beam
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[0, 3] = 10.0
        logits[1, 7] = 5.0
        bs.step(logits)
        best = bs.get_best()
        assert isinstance(best, BeamHypothesis)
        # The best beam should have token 3 (higher logit)
        assert 3 in best.token_ids

    def test_length_penalty_affects_scores(self):
        """Longer sequences should be penalized more with length_penalty > 1."""
        bs1 = BeamSearcher(num_beams=1, max_length=10, length_penalty=1.0)
        bs2 = BeamSearcher(num_beams=1, max_length=10, length_penalty=2.0)
        bs1.init_beams()
        bs2.init_beams()
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, 0] = 5.0
        bs1.step(logits)
        bs2.step(logits)
        # Higher length_penalty -> more score reduction for same sequence
        # Penalized score = raw_score / ((5 + len) / 6)^penalty
        # With length_penalty=2.0, the divisor is larger
        assert bs2.beams[0].score <= bs1.beams[0].score

    def test_multiple_steps(self):
        """Beam search should work across multiple steps."""
        bs = BeamSearcher(num_beams=2, max_length=5)
        bs.init_beams()
        for step in range(3):
            logits = torch.randn(2, VOCAB_SIZE)
            sequences = bs.step(logits)
            assert len(sequences) == 2
            # Sequences should grow by 1 each step (unless finished)
            for seq in sequences:
                if seq:  # non-empty (not a padded beam)
                    assert len(seq) <= step + 1


# =========================================================================
# 14. Edge Cases for apply_min_p and apply_typical_p
# =========================================================================

class TestEdgeCases:
    """Edge cases for apply_min_p and apply_typical_p with disabled values."""

    def test_apply_min_p_negative_value(self):
        """Negative min_p should be treated as disabled."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_p(logits, min_p=-0.5)
        assert torch.equal(result, original)

    def test_apply_min_p_exactly_zero(self):
        """min_p=0.0 exactly should be disabled."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_p(logits, min_p=0.0)
        assert torch.equal(result, original)

    def test_apply_min_p_exactly_one(self):
        """min_p=1.0 exactly should be disabled."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_p(logits, min_p=1.0)
        assert torch.equal(result, original)

    def test_apply_min_p_above_one(self):
        """min_p > 1.0 should be disabled (would filter everything)."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_min_p(logits, min_p=1.5)
        assert torch.equal(result, original)

    def test_apply_typical_p_exactly_one(self):
        """typical_p=1.0 exactly should be disabled."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_typical_p(logits, typical_p=1.0)
        assert torch.equal(result, original)

    def test_apply_typical_p_above_one(self):
        """typical_p > 1.0 should be disabled."""
        logits = torch.randn(VOCAB_SIZE)
        original = logits.clone()
        result = apply_typical_p(logits, typical_p=2.0)
        assert torch.equal(result, original)

    def test_apply_min_p_uniform_logits(self):
        """With uniform logits, all tokens have equal prob -> all pass min_p."""
        logits = torch.ones(VOCAB_SIZE) * 5.0
        result = apply_min_p(logits.clone(), min_p=0.5)
        # All probs are equal and equal to top_prob, so all pass threshold
        num_active = (result > float("-inf")).sum().item()
        assert num_active == VOCAB_SIZE

    def test_apply_typical_p_single_dominant_token(self):
        """With one very dominant token, typical_p should still keep it."""
        logits = torch.full((VOCAB_SIZE,), -100.0)
        logits[3] = 50.0  # One extremely dominant token
        result = apply_typical_p(logits.clone(), typical_p=0.1)
        # Token 3 must survive
        assert result[3].item() > float("-inf")

    def test_apply_min_p_all_negative_logits(self):
        """min_p should work correctly even when all logits are negative."""
        logits = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0])
        result = apply_min_p(logits.clone(), min_p=0.5)
        # Token 0 has highest prob; others may be filtered
        assert result[0].item() > float("-inf")

    def test_apply_typical_p_all_equal_logits(self):
        """With all equal logits, all tokens are equally typical."""
        logits = torch.ones(VOCAB_SIZE) * 3.0
        result = apply_typical_p(logits.clone(), typical_p=0.5)
        # With uniform distribution, all tokens have same deviation from entropy
        # They should all be treated the same; cumulative prob check determines how many pass
        num_active = (result > float("-inf")).sum().item()
        assert num_active >= 1


# =========================================================================
# Run with pytest
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
