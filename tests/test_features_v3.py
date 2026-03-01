"""
vllm-i64 :: Test Features V3

Tests for the third batch of features:
  - Seed parameter (reproducible generation)
  - Logit bias (boost/suppress tokens)
  - Frequency penalty + presence penalty
  - Batch completions endpoint
  - Graceful shutdown
  - Request priority queue
  - Model info endpoint
  - Latency tracker (p50/p95/p99)
  - Structured request logging

Run:
    python -m pytest tests/test_features_v3.py -v

INL - 2025
"""

import pytest
import torch
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.core.sampling import (
    SamplingParams, sample_batch, sample_batch_with_logprobs,
    apply_logit_bias, apply_frequency_presence_penalty_batch,
)


# =====================================================================
# 1. Seed Parameter
# =====================================================================

class TestSeedParameter:
    """Test seed for reproducible generation."""

    def test_seed_produces_same_result(self):
        """Same seed → same sampled token."""
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, seed=42)
        t1 = sample_batch(logits.clone(), params)
        t2 = sample_batch(logits.clone(), params)
        assert t1.item() == t2.item()

    def test_different_seeds_differ(self):
        """Different seeds → different tokens (with high probability)."""
        logits = torch.randn(1, 10000)  # Large vocab for high entropy
        params1 = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, seed=1)
        params2 = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, seed=999)
        t1 = sample_batch(logits.clone(), params1)
        t2 = sample_batch(logits.clone(), params2)
        # Very unlikely to be the same with 10000 tokens
        assert t1.item() != t2.item()

    def test_no_seed_is_default(self):
        """No seed → no crash, sampling works normally."""
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=0.8, seed=None)
        token = sample_batch(logits, params)
        assert 0 <= token.item() < 100

    def test_seed_greedy_no_effect(self):
        """Seed doesn't matter for greedy (temp=0) — always same result."""
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=0.0, seed=42)
        t1 = sample_batch(logits.clone(), params)
        t2 = sample_batch(logits.clone(), params)
        assert t1.item() == t2.item()

    def test_seed_with_logprobs(self):
        """Seed works with sample_batch_with_logprobs."""
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, seed=42, logprobs=3)
        out1 = sample_batch_with_logprobs(logits.clone(), params)
        out2 = sample_batch_with_logprobs(logits.clone(), params)
        assert out1.token_ids.item() == out2.token_ids.item()

    def test_seed_in_sampling_params(self):
        """SamplingParams has seed field."""
        params = SamplingParams(seed=123)
        assert params.seed == 123


# =====================================================================
# 2. Logit Bias
# =====================================================================

class TestLogitBias:
    """Test logit bias for boosting/suppressing tokens."""

    def test_apply_logit_bias(self):
        """apply_logit_bias modifies specific token logits."""
        logits = torch.zeros(1, 10)
        apply_logit_bias(logits, {3: 100.0, 7: -100.0})
        assert logits[0, 3].item() == 100.0
        assert logits[0, 7].item() == -100.0
        assert logits[0, 0].item() == 0.0

    def test_logit_bias_forces_token(self):
        """Large positive bias forces greedy selection of that token."""
        logits = torch.zeros(1, 100)
        params = SamplingParams(temperature=0.0, logit_bias={42: 1000.0})
        token = sample_batch(logits, params)
        assert token.item() == 42

    def test_logit_bias_suppresses_token(self):
        """Large negative bias prevents selection."""
        logits = torch.zeros(1, 10)
        logits[0, 5] = 10.0  # Token 5 would normally be selected
        params = SamplingParams(temperature=0.0, logit_bias={5: -1000.0})
        token = sample_batch(logits, params)
        assert token.item() != 5

    def test_logit_bias_out_of_range_ignored(self):
        """Token IDs outside vocab range are silently ignored."""
        logits = torch.zeros(1, 10)
        apply_logit_bias(logits, {999: 100.0})  # Out of range
        assert logits.sum().item() == 0.0

    def test_logit_bias_empty_dict_noop(self):
        """Empty logit_bias is a no-op."""
        logits = torch.randn(1, 10)
        original = logits.clone()
        apply_logit_bias(logits, {})
        assert torch.equal(logits, original)

    def test_logit_bias_in_sampling_params(self):
        """SamplingParams accepts logit_bias."""
        params = SamplingParams(logit_bias={1: 5.0, 2: -3.0})
        assert params.logit_bias == {1: 5.0, 2: -3.0}


# =====================================================================
# 3. Frequency Penalty + Presence Penalty
# =====================================================================

class TestFrequencyPresencePenalty:
    """Test OpenAI-style frequency and presence penalties."""

    def test_frequency_penalty_reduces_repeated_tokens(self):
        """Frequency penalty reduces logits of tokens proportional to count."""
        logits = torch.zeros(1, 10)
        past = [[3, 3, 3, 7]]  # Token 3 appears 3 times, token 7 once
        apply_frequency_presence_penalty_batch(logits, past, frequency_penalty=1.0, presence_penalty=0.0, vocab_size=10)
        assert logits[0, 3].item() == pytest.approx(-3.0)
        assert logits[0, 7].item() == pytest.approx(-1.0)
        assert logits[0, 0].item() == 0.0

    def test_presence_penalty_binary(self):
        """Presence penalty is binary — penalizes all seen tokens equally."""
        logits = torch.zeros(1, 10)
        past = [[3, 3, 3, 7]]
        apply_frequency_presence_penalty_batch(logits, past, frequency_penalty=0.0, presence_penalty=1.0, vocab_size=10)
        assert logits[0, 3].item() == pytest.approx(-1.0)  # Seen → -1
        assert logits[0, 7].item() == pytest.approx(-1.0)  # Seen → -1
        assert logits[0, 0].item() == 0.0  # Not seen → 0

    def test_combined_penalties(self):
        """Frequency + presence penalties stack."""
        logits = torch.zeros(1, 10)
        past = [[5, 5]]
        apply_frequency_presence_penalty_batch(logits, past, frequency_penalty=0.5, presence_penalty=0.5, vocab_size=10)
        # Token 5: -0.5*2 (freq) - 0.5*1 (presence) = -1.5
        assert logits[0, 5].item() == pytest.approx(-1.5)

    def test_zero_penalties_noop(self):
        """Zero penalties don't modify logits."""
        logits = torch.randn(1, 10)
        original = logits.clone()
        apply_frequency_presence_penalty_batch(logits, [[1, 2, 3]], 0.0, 0.0, 10)
        assert torch.equal(logits, original)

    def test_empty_past_noop(self):
        """Empty past tokens = no effect."""
        logits = torch.randn(1, 10)
        original = logits.clone()
        apply_frequency_presence_penalty_batch(logits, [[]], 1.0, 1.0, 10)
        assert torch.equal(logits, original)

    def test_frequency_penalty_in_sample_batch(self):
        """Frequency penalty integrated in sample_batch."""
        logits = torch.zeros(1, 10)
        logits[0, 3] = 5.0  # Would be selected without penalty
        logits[0, 0] = 4.9  # Close second
        params = SamplingParams(temperature=0.0, frequency_penalty=2.0)
        past = [[3, 3, 3]]  # Token 3 repeated 3 times
        token = sample_batch(logits, params, past_tokens_list=past)
        # With freq_penalty=2.0 and 3 occurrences, logit[3] -= 6.0 → becomes -1.0
        # Token 0 at 4.9 should win
        assert token.item() == 0

    def test_penalties_in_sampling_params(self):
        """SamplingParams has frequency/presence penalty fields."""
        params = SamplingParams(frequency_penalty=0.5, presence_penalty=0.3)
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3


# =====================================================================
# 4. CompletionRequest validation for new fields
# =====================================================================

from vllm_i64.api.server import CompletionRequest


class TestCompletionRequestValidation:
    """Test validation of new CompletionRequest fields."""

    def test_valid_new_fields(self):
        """Valid request with all new fields."""
        req = CompletionRequest(
            prompt="hello",
            seed=42,
            logit_bias={"1": 5.0, "2": -3.0},
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        assert req.validate() is None

    def test_frequency_penalty_range(self):
        """frequency_penalty must be in [-2, 2]."""
        req = CompletionRequest(prompt="hello", frequency_penalty=3.0)
        assert "frequency_penalty" in req.validate()

    def test_presence_penalty_range(self):
        """presence_penalty must be in [-2, 2]."""
        req = CompletionRequest(prompt="hello", presence_penalty=-3.0)
        assert "presence_penalty" in req.validate()

    def test_logit_bias_value_range(self):
        """logit_bias values must be in [-100, 100]."""
        req = CompletionRequest(prompt="hello", logit_bias={"1": 200.0})
        assert "logit_bias" in req.validate()

    def test_logit_bias_invalid_key(self):
        """logit_bias keys must be numeric strings."""
        req = CompletionRequest(prompt="hello", logit_bias={"abc": 1.0})
        assert "logit_bias" in req.validate()

    def test_to_sampling_params_includes_new_fields(self):
        """to_sampling_params() converts all new fields."""
        req = CompletionRequest(
            prompt="hello",
            seed=42,
            logit_bias={"5": 2.0},
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        params = req.to_sampling_params()
        assert params.seed == 42
        assert params.logit_bias == {5: 2.0}
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3


# =====================================================================
# 5. Latency Tracker
# =====================================================================

from vllm_i64.api.server import LatencyTracker


class TestLatencyTracker:
    """Test latency percentile tracking."""

    def test_empty_tracker(self):
        """Empty tracker returns zeros."""
        tracker = LatencyTracker()
        p = tracker.percentiles()
        assert p["count"] == 0
        assert p["p50_ms"] == 0.0

    def test_record_and_percentiles(self):
        """Recording latencies and computing percentiles."""
        tracker = LatencyTracker()
        for i in range(100):
            tracker.record("/v1/completions", float(i))
        p = tracker.percentiles()
        assert p["count"] == 100
        assert p["p50_ms"] == 50.0
        assert p["p95_ms"] == 95.0
        assert p["p99_ms"] == 99.0

    def test_per_endpoint_tracking(self):
        """Different endpoints tracked separately."""
        tracker = LatencyTracker()
        for i in range(10):
            tracker.record("/v1/completions", 10.0)
            tracker.record("/v1/chat/completions", 50.0)

        p_comp = tracker.percentiles("/v1/completions")
        p_chat = tracker.percentiles("/v1/chat/completions")
        assert p_comp["p50_ms"] == 10.0
        assert p_chat["p50_ms"] == 50.0

    def test_max_window(self):
        """Tracker respects max_window size."""
        tracker = LatencyTracker(max_window=10)
        for i in range(100):
            tracker.record("/test", float(i))
        p = tracker.percentiles()
        assert p["count"] == 10  # Only last 10

    def test_get_all_endpoints(self):
        """get_all_endpoints returns overall + per-endpoint."""
        tracker = LatencyTracker()
        tracker.record("/a", 1.0)
        tracker.record("/b", 2.0)
        all_eps = tracker.get_all_endpoints()
        assert "overall" in all_eps
        assert "/a" in all_eps
        assert "/b" in all_eps


# =====================================================================
# 6. Request Logger
# =====================================================================

from vllm_i64.api.server import RequestLogger


class TestRequestLogger:
    """Test structured request logging."""

    def test_log_request(self):
        """Logging a request stores entry."""
        logger = RequestLogger()
        logger.log_request("/v1/completions", 200, 50.0, prompt_tokens=10, completion_tokens=20)
        entries = logger.get_recent()
        assert len(entries) == 1
        assert entries[0]["endpoint"] == "/v1/completions"
        assert entries[0]["status"] == 200
        assert entries[0]["latency_ms"] == 50.0
        assert entries[0]["prompt_tokens"] == 10

    def test_log_with_error(self):
        """Log entry includes error field when provided."""
        logger = RequestLogger()
        logger.log_request("/v1/completions", 500, 10.0, error="OOM")
        entries = logger.get_recent()
        assert entries[0]["error"] == "OOM"

    def test_api_key_truncated(self):
        """API key is truncated for privacy."""
        logger = RequestLogger()
        logger.log_request("/test", 200, 1.0, api_key="sk-long-secret-key-12345")
        entries = logger.get_recent()
        assert entries[0]["api_key"] == "sk-long-..."

    def test_max_log_size(self):
        """Logger respects max_log limit."""
        logger = RequestLogger(max_log=5)
        for i in range(10):
            logger.log_request("/test", 200, float(i))
        entries = logger.get_recent(100)
        assert len(entries) == 5

    def test_disabled_logger(self):
        """Disabled logger doesn't store entries."""
        logger = RequestLogger(enabled=False)
        logger.log_request("/test", 200, 1.0)
        assert len(logger.get_recent()) == 0

    def test_get_recent_limit(self):
        """get_recent(n) returns at most n entries."""
        logger = RequestLogger()
        for i in range(10):
            logger.log_request("/test", 200, float(i))
        entries = logger.get_recent(3)
        assert len(entries) == 3


# =====================================================================
# 7. Priority Manager
# =====================================================================

from vllm_i64.api.server import PriorityManager


class TestPriorityManager:
    """Test API key priority management."""

    def test_default_priority_zero(self):
        """Unknown keys have priority 0."""
        mgr = PriorityManager()
        assert mgr.get_priority("unknown-key") == 0

    def test_set_and_get_priority(self):
        """Set and retrieve API key priority."""
        mgr = PriorityManager()
        mgr.set_priority("key-vip", 10)
        assert mgr.get_priority("key-vip") == 10

    def test_request_priority_override(self):
        """Request-level priority overrides key priority if higher."""
        mgr = PriorityManager()
        mgr.set_priority("key-a", 5)
        assert mgr.get_priority("key-a", request_priority=10) == 10
        assert mgr.get_priority("key-a", request_priority=2) == 5

    def test_none_api_key(self):
        """None API key returns request priority."""
        mgr = PriorityManager()
        assert mgr.get_priority(None, request_priority=3) == 3

    def test_get_all(self):
        """get_all returns all registered priorities."""
        mgr = PriorityManager()
        mgr.set_priority("a", 1)
        mgr.set_priority("b", 2)
        all_p = mgr.get_all()
        assert all_p == {"a": 1, "b": 2}


# =====================================================================
# 8. Server Integration — new routes
# =====================================================================

from vllm_i64.api.server import I64Server


class TestServerV3Integration:
    """Integration tests for V3 server features."""

    def _make_server(self, **kwargs):
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        return I64Server(engine=engine, **kwargs)

    def test_new_routes_registered(self):
        """All new V3 routes are registered."""
        server = self._make_server()
        app = server.create_app()
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, 'resource')]
        assert "/v1/batch" in routes
        assert "/v1/models/{model_id}" in routes
        assert "/v1/metrics" in routes
        assert "/v1/logs" in routes
        assert "/v1/priority" in routes

    def test_server_has_latency_tracker(self):
        """Server has latency tracker."""
        server = self._make_server()
        assert isinstance(server._latency_tracker, LatencyTracker)

    def test_server_has_request_logger(self):
        """Server has request logger."""
        server = self._make_server()
        assert isinstance(server._request_logger, RequestLogger)

    def test_server_has_priority_manager(self):
        """Server has priority manager."""
        server = self._make_server()
        assert isinstance(server._priority_manager, PriorityManager)

    def test_server_shutting_down_flag(self):
        """Server has _shutting_down flag."""
        server = self._make_server()
        assert server._shutting_down is False

    def test_completion_request_priority(self):
        """CompletionRequest has priority field."""
        req = CompletionRequest(prompt="hello", priority=5)
        assert req.priority == 5


# =====================================================================
# 9. Model Info (unit-level)
# =====================================================================

class TestModelInfo:
    """Test model info endpoint construction."""

    def test_model_info_no_model(self):
        """Model info works even without a loaded model."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine, model_name="test-model")
        # Server should have model_name
        assert server.model_name == "test-model"

    def test_model_info_with_model(self):
        """Model info includes config when model is loaded."""
        import torch.nn as nn

        class MockConfig:
            num_experts = 4
            vocab_size = 100
            hidden_size = 32
            num_hidden_layers = 2
            num_attention_heads = 4
            num_key_value_heads = 2
            head_dim = 8
            eos_token_id = 0

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.embed_tokens = nn.Embedding(100, 32)
                self.norm = None

            def forward(self, **kwargs):
                return torch.randn(1, 100)

        model = MockModel()
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=model, num_experts=4, vocab_size=100, device="cpu")

        # Verify engine has model with config
        assert hasattr(engine.model, 'config')
        assert engine.model.config.num_experts == 4
        assert engine.model.config.hidden_size == 32
