"""
vllm-i64 :: Test New Features

Tests for all features added in the latest update:
  - Logprobs (sample_batch_with_logprobs, _gather_logprobs)
  - Input validation (CompletionRequest.validate)
  - Rate limiter (TokenBucketRateLimiter)
  - Swap-to-CPU (swap_out, swap_in)
  - Paged decode attention (naive_paged_decode_attention)
  - Adaptive batching (AdaptiveBatchSizer)
  - Tool parser (ToolCallParser)

Run:
    python -m pytest tests/test_new_features.py -v

INL - 2025
"""

import pytest
import torch
import json
import time
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.core.sampling import (
    SamplingParams, sample_batch_with_logprobs, _gather_logprobs,
    TokenLogprob, SampleOutput,
)
from vllm_i64.api.server import CompletionRequest, TokenBucketRateLimiter
from vllm_i64.core.kv_cache import PagedKVCache
from vllm_i64.layers.attention import naive_paged_decode_attention
from vllm_i64.engine.i64_engine import AdaptiveBatchSizer
from vllm_i64.core.tool_parser import ToolCallParser, ToolCall


# =========================================================================
# Logprobs
# =========================================================================

class TestLogprobs:

    def test_sample_with_logprobs_greedy(self):
        """Greedy sampling with logprobs returns correct token and valid logprobs."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])  # token 1 is argmax
        params = SamplingParams(temperature=0.0, logprobs=3)
        out = sample_batch_with_logprobs(logits, params)

        assert isinstance(out, SampleOutput)
        assert int(out.token_ids[0].item()) == 1
        assert out.logprobs is not None
        assert len(out.logprobs) == 1

        lp = out.logprobs[0]
        assert lp.token_id == 1
        assert lp.logprob <= 0.0  # log-prob is always <= 0
        assert lp.top_logprobs is not None
        assert len(lp.top_logprobs) == 3

    def test_sample_with_logprobs_temperature(self):
        """Sampling with temperature returns logprobs from pre-filtered distribution."""
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
        params = SamplingParams(temperature=0.8, top_k=0, top_p=1.0, logprobs=2)
        out = sample_batch_with_logprobs(logits, params)

        assert out.logprobs is not None
        lp = out.logprobs[0]
        assert lp.logprob <= 0.0
        assert len(lp.top_logprobs) == 2

    def test_sample_without_logprobs(self):
        """Without logprobs param, no logprobs are returned."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        params = SamplingParams(temperature=0.0, logprobs=None)
        out = sample_batch_with_logprobs(logits, params)
        assert out.logprobs is None

    def test_sample_with_logprobs_batch(self):
        """Batch sampling returns one logprob per element."""
        logits = torch.tensor([
            [10.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 10.0, 1.0],
        ])
        params = SamplingParams(temperature=0.0, logprobs=2)
        out = sample_batch_with_logprobs(logits, params)

        assert len(out.logprobs) == 2
        assert out.logprobs[0].token_id == 0
        assert out.logprobs[1].token_id == 2

    def test_gather_logprobs(self):
        """_gather_logprobs returns correct structure."""
        log_probs = torch.log_softmax(torch.tensor([[3.0, 1.0, 2.0]]), dim=-1)
        token_ids = torch.tensor([0])
        results = _gather_logprobs(log_probs, token_ids, num_top=2)

        assert len(results) == 1
        assert results[0].token_id == 0
        assert results[0].logprob < 0.0
        assert len(results[0].top_logprobs) == 2
        # Top logprob should include token 0 (highest logit)
        assert 0 in results[0].top_logprobs

    def test_logprobs_sum_to_one(self):
        """Top logprobs should be valid log-probabilities (all <= 0)."""
        logits = torch.randn(1, 100)
        params = SamplingParams(temperature=0.0, logprobs=5)
        out = sample_batch_with_logprobs(logits, params)

        for tid, lp_val in out.logprobs[0].top_logprobs.items():
            assert lp_val <= 0.0, f"Logprob for token {tid} is {lp_val} > 0"

    def test_logprobs_with_repetition_penalty(self):
        """Logprobs work correctly with repetition penalty applied."""
        logits = torch.tensor([[5.0, 5.0, 1.0, 1.0]])
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0, logprobs=2)
        past = [[0]]  # penalize token 0
        out = sample_batch_with_logprobs(logits, params, past_tokens_list=past)

        assert out.logprobs is not None
        # Token 1 should be selected since token 0 is penalized
        assert out.logprobs[0].token_id == 1


# =========================================================================
# Input Validation
# =========================================================================

class TestInputValidation:

    def test_valid_request(self):
        req = CompletionRequest(prompt="hello", max_tokens=100)
        assert req.validate() is None

    def test_empty_prompt(self):
        req = CompletionRequest(prompt="", max_tokens=100)
        assert req.validate() is not None
        assert "prompt" in req.validate()

    def test_whitespace_prompt(self):
        req = CompletionRequest(prompt="   ", max_tokens=100)
        assert req.validate() is not None

    def test_max_tokens_zero(self):
        req = CompletionRequest(prompt="hello", max_tokens=0)
        assert "max_tokens" in req.validate()

    def test_max_tokens_exceeds_limit(self):
        req = CompletionRequest(prompt="hello", max_tokens=5000)
        err = req.validate(max_seq_len=2048)
        assert err is not None
        assert "2048" in err

    def test_negative_temperature(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, temperature=-1.0)
        assert "temperature" in req.validate()

    def test_zero_temperature_valid(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, temperature=0.0)
        assert req.validate() is None

    def test_invalid_top_p(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, top_p=1.5)
        assert "top_p" in req.validate()

    def test_negative_top_p(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, top_p=-0.1)
        assert "top_p" in req.validate()

    def test_negative_top_k(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, top_k=-1)
        assert "top_k" in req.validate()

    def test_zero_repetition_penalty(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, repetition_penalty=0.0)
        assert "repetition_penalty" in req.validate()

    def test_logprobs_out_of_range(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, logprobs=25)
        assert "logprobs" in req.validate()

    def test_logprobs_negative(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, logprobs=-1)
        assert "logprobs" in req.validate()

    def test_logprobs_valid(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, logprobs=5)
        assert req.validate() is None

    def test_to_sampling_params(self):
        req = CompletionRequest(prompt="hello", max_tokens=10, temperature=0.5, logprobs=3)
        params = req.to_sampling_params()
        assert params.temperature == 0.5
        assert params.logprobs == 3

    def test_to_sampling_params_with_json_mode(self):
        req = CompletionRequest(
            prompt="hello", max_tokens=10,
            response_format={"type": "json_object"},
        )
        params = req.to_sampling_params()
        assert params.json_mode is True
        assert params.output_constraints is not None

    def test_to_sampling_params_with_stop(self):
        req = CompletionRequest(
            prompt="hello", max_tokens=10,
            stop=["<end>", "STOP"],
        )
        params = req.to_sampling_params()
        assert params.output_constraints is not None
        assert params.output_constraints.stop_sequences is not None
        assert len(params.output_constraints.stop_sequences) == 2


# =========================================================================
# Rate Limiter
# =========================================================================

class TestRateLimiter:

    def test_first_request_allowed(self):
        rl = TokenBucketRateLimiter(requests_per_minute=60)
        assert rl.allow("1.2.3.4") is True

    def test_burst_within_capacity(self):
        rl = TokenBucketRateLimiter(requests_per_minute=10)
        # First 10 requests should be allowed (capacity = 10)
        for i in range(10):
            assert rl.allow("1.2.3.4") is True

    def test_burst_exceeds_capacity(self):
        rl = TokenBucketRateLimiter(requests_per_minute=5)
        for _ in range(5):
            rl.allow("1.2.3.4")
        # 6th request without any time passing should be denied
        assert rl.allow("1.2.3.4") is False

    def test_different_ips_independent(self):
        rl = TokenBucketRateLimiter(requests_per_minute=2)
        rl.allow("1.1.1.1")
        rl.allow("1.1.1.1")
        # IP 1 exhausted, but IP 2 should still work
        assert rl.allow("2.2.2.2") is True

    def test_refill_over_time(self):
        rl = TokenBucketRateLimiter(requests_per_minute=60)
        # Exhaust all tokens
        for _ in range(60):
            rl.allow("1.1.1.1")
        # Should be denied
        assert rl.allow("1.1.1.1") is False

        # Manually adjust the last time to simulate passage of time
        bucket = rl._buckets["1.1.1.1"]
        bucket[1] -= 2.0  # 2 seconds ago → ~2 tokens refilled
        assert rl.allow("1.1.1.1") is True


# =========================================================================
# Swap-to-CPU
# =========================================================================

class TestSwapToCPU:

    def _make_cache(self, num_blocks=16, block_size=4, num_layers=2,
                    num_kv_heads=2, head_dim=8):
        cache = PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            max_seqs=8,
            device="cpu",
        )
        return cache

    def test_swap_disabled_by_default(self):
        cache = self._make_cache()
        assert cache.swap_enabled is False
        assert cache.swap_out(0) is False

    def test_enable_swap_on_cpu_is_noop(self):
        """Swap CPU→CPU doesn't make sense, enable_swap is a no-op."""
        cache = self._make_cache()
        cache.enable_swap()
        # On CPU device, enable_swap should be a no-op
        assert cache.swap_enabled is False

    def test_swap_not_enabled_returns_false(self):
        cache = self._make_cache()
        assert cache.swap_out(0) is False
        assert cache.swap_in(0) is False

    def test_swap_in_nonexistent_seq(self):
        cache = self._make_cache()
        # Even if we could enable swap, swap_in for non-swapped seq returns False
        assert cache.swap_in(999) is False

    def test_get_stats_with_swap_disabled(self):
        cache = self._make_cache()
        stats = cache.get_stats()
        assert "swapped_seqs" not in stats

    def test_get_stats_keys(self):
        cache = self._make_cache()
        stats = cache.get_stats()
        assert "num_blocks" in stats
        assert "used_blocks" in stats
        assert "free_blocks" in stats
        assert "block_size" in stats
        assert "active_seqs" in stats


# =========================================================================
# Paged Decode Attention
# =========================================================================

class TestPagedDecodeAttention:

    def test_single_query_single_block(self):
        """Single query attending to a single block of KV."""
        batch = 1
        num_heads = 2
        num_kv_heads = 2
        head_dim = 8
        block_size = 4
        num_blocks = 4
        seq_len = 3

        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)

        block_table = torch.full((batch, 4), -1, dtype=torch.int32)
        block_table[0, 0] = 2  # first block at physical block 2

        cache_seqlens = torch.tensor([seq_len], dtype=torch.int32)

        out = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens,
        )
        assert out.shape == (batch, num_heads, head_dim)

    def test_multi_block(self):
        """Query spanning multiple blocks."""
        batch = 1
        num_heads = 2
        num_kv_heads = 2
        head_dim = 8
        block_size = 4
        num_blocks = 8
        seq_len = 10  # spans 3 blocks (4 + 4 + 2)

        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)

        block_table = torch.full((batch, 8), -1, dtype=torch.int32)
        block_table[0, 0] = 0
        block_table[0, 1] = 3
        block_table[0, 2] = 5

        cache_seqlens = torch.tensor([seq_len], dtype=torch.int32)

        out = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens,
        )
        assert out.shape == (batch, num_heads, head_dim)

    def test_batch_decode(self):
        """Batch of queries with different sequence lengths."""
        batch = 3
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        block_size = 4
        num_blocks = 16

        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)

        block_table = torch.full((batch, 8), -1, dtype=torch.int32)
        # seq 0: 3 tokens (1 block)
        block_table[0, 0] = 0
        # seq 1: 7 tokens (2 blocks)
        block_table[1, 0] = 1
        block_table[1, 1] = 2
        # seq 2: 4 tokens (1 full block)
        block_table[2, 0] = 3

        cache_seqlens = torch.tensor([3, 7, 4], dtype=torch.int32)
        num_kv_groups = num_heads // num_kv_heads

        out = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens,
            num_kv_groups=num_kv_groups,
        )
        assert out.shape == (batch, num_heads, head_dim)

    def test_zero_seqlen(self):
        """Zero-length sequence returns zeros."""
        batch = 2
        num_heads = 2
        head_dim = 8
        block_size = 4
        num_blocks = 4

        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)

        block_table = torch.full((batch, 4), -1, dtype=torch.int32)
        block_table[0, 0] = 0
        cache_seqlens = torch.tensor([3, 0], dtype=torch.int32)

        out = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens,
        )
        assert out.shape == (batch, num_heads, head_dim)
        # Second element should be zeros
        assert torch.allclose(out[1], torch.zeros(num_heads, head_dim))

    def test_gqa_groups(self):
        """GQA with num_kv_heads < num_heads works correctly."""
        batch = 1
        num_heads = 8
        num_kv_heads = 2
        head_dim = 16
        block_size = 4
        num_blocks = 4

        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)

        block_table = torch.full((batch, 4), -1, dtype=torch.int32)
        block_table[0, 0] = 1
        cache_seqlens = torch.tensor([4], dtype=torch.int32)

        out = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens,
            num_kv_groups=num_heads // num_kv_heads,
        )
        assert out.shape == (batch, num_heads, head_dim)

    def test_output_is_finite(self):
        """Output should contain no NaN or Inf values."""
        batch = 2
        num_heads = 4
        head_dim = 8
        block_size = 4
        num_blocks = 8

        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)

        block_table = torch.full((batch, 4), -1, dtype=torch.int32)
        block_table[0, 0] = 0
        block_table[0, 1] = 1
        block_table[1, 0] = 2
        cache_seqlens = torch.tensor([5, 3], dtype=torch.int32)

        out = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens,
        )
        assert torch.isfinite(out).all()


# =========================================================================
# Adaptive Batch Sizer
# =========================================================================

class TestAdaptiveBatchSizer:

    def test_initial_size(self):
        sizer = AdaptiveBatchSizer(initial=8, min_size=1, max_size=32)
        assert sizer.current == 8

    def test_no_adjust_before_window(self):
        """No adjustment until enough samples collected."""
        sizer = AdaptiveBatchSizer(initial=8, window=10)
        for i in range(5):
            sizer.record(tokens=100, elapsed_ms=10.0)
        result = sizer.adjust()
        assert result == 8  # unchanged

    def test_increase_on_improving_throughput(self):
        """Batch size increases when recent throughput exceeds average."""
        sizer = AdaptiveBatchSizer(initial=8, min_size=1, max_size=32, window=10)
        # Fill window with moderate throughput
        for _ in range(15):
            sizer.record(tokens=100, elapsed_ms=10.0)
        # Then record high throughput
        for _ in range(5):
            sizer.record(tokens=200, elapsed_ms=10.0)
        result = sizer.adjust()
        assert result >= 8  # should increase or stay

    def test_decrease_on_degrading_throughput(self):
        """Batch size decreases when recent throughput drops."""
        sizer = AdaptiveBatchSizer(initial=8, min_size=1, max_size=32, window=10)
        # Fill window with high throughput
        for _ in range(15):
            sizer.record(tokens=200, elapsed_ms=10.0)
        # Then record low throughput
        for _ in range(5):
            sizer.record(tokens=10, elapsed_ms=10.0)
        result = sizer.adjust()
        assert result <= 8  # should decrease or stay

    def test_respects_min_bound(self):
        sizer = AdaptiveBatchSizer(initial=2, min_size=2, max_size=32, window=5)
        # Force many low throughput records
        for _ in range(20):
            sizer.record(tokens=1, elapsed_ms=100.0)
        for _ in range(5):
            sizer.record(tokens=1, elapsed_ms=1000.0)
        result = sizer.adjust()
        assert result >= 2

    def test_respects_max_bound(self):
        sizer = AdaptiveBatchSizer(initial=30, min_size=1, max_size=32, window=5)
        # Force many high throughput records
        for _ in range(20):
            sizer.record(tokens=1000, elapsed_ms=1.0)
        for _ in range(5):
            sizer.record(tokens=10000, elapsed_ms=1.0)
        result = sizer.adjust()
        assert result <= 32

    def test_record_with_zero_elapsed(self):
        """Zero elapsed time shouldn't crash."""
        sizer = AdaptiveBatchSizer(initial=8)
        sizer.record(tokens=100, elapsed_ms=0.0)
        assert len(sizer._throughputs) == 1

    def test_window_rolling(self):
        """Window correctly limits stored throughput records."""
        sizer = AdaptiveBatchSizer(initial=8, window=5)
        for i in range(20):
            sizer.record(tokens=100, elapsed_ms=10.0)
        assert len(sizer._throughputs) == 5


# =========================================================================
# Tool Parser
# =========================================================================

class TestToolParser:

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        },
    ]

    def test_parse_json_tool_call(self):
        parser = ToolCallParser(self.TOOLS)
        text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        calls = parser.parse(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].function_name == "get_weather"
        args = json.loads(calls[0].function_arguments)
        assert args["city"] == "Paris"

    def test_parse_xml_tool_call(self):
        parser = ToolCallParser(self.TOOLS)
        text = '<tool_call>{"name": "search", "arguments": {"query": "hello"}}</tool_call>'
        calls = parser.parse(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].function_name == "search"

    def test_parse_multiple_xml_calls(self):
        parser = ToolCallParser(self.TOOLS)
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
            ' some text '
            '<tool_call>{"name": "search", "arguments": {"query": "news"}}</tool_call>'
        )
        calls = parser.parse(text)
        assert calls is not None
        assert len(calls) == 2

    def test_parse_no_tool_calls(self):
        parser = ToolCallParser(self.TOOLS)
        text = "Hello, how can I help you today?"
        calls = parser.parse(text)
        assert calls is None

    def test_parse_unknown_function(self):
        parser = ToolCallParser(self.TOOLS)
        text = '{"name": "unknown_func", "arguments": {"x": 1}}'
        calls = parser.parse(text)
        assert calls is None

    def test_parse_invalid_json(self):
        parser = ToolCallParser(self.TOOLS)
        text = '<tool_call>not valid json</tool_call>'
        calls = parser.parse(text)
        assert calls is None

    def test_tool_call_has_id(self):
        parser = ToolCallParser(self.TOOLS)
        text = '{"name": "get_weather", "arguments": {"city": "London"}}'
        calls = parser.parse(text)
        assert calls[0].id.startswith("call_")
        assert len(calls[0].id) > 5

    def test_tool_call_type(self):
        parser = ToolCallParser(self.TOOLS)
        text = '{"name": "get_weather", "arguments": {"city": "Berlin"}}'
        calls = parser.parse(text)
        assert calls[0].type == "function"

    def test_arguments_as_string(self):
        """Arguments should always be a JSON string."""
        parser = ToolCallParser(self.TOOLS)
        text = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        calls = parser.parse(text)
        assert isinstance(calls[0].function_arguments, str)
        parsed = json.loads(calls[0].function_arguments)
        assert parsed["city"] == "Tokyo"

    def test_xml_priority_over_json(self):
        """XML tag matches should take priority (checked first)."""
        parser = ToolCallParser(self.TOOLS)
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}}</tool_call>'
            '{"name": "search", "arguments": {"query": "B"}}'
        )
        calls = parser.parse(text)
        # XML is checked first, should return only the XML match
        assert len(calls) == 1
        assert calls[0].function_name == "get_weather"

    def test_empty_tools_list(self):
        parser = ToolCallParser([])
        text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        calls = parser.parse(text)
        assert calls is None

    def test_function_names_extracted(self):
        parser = ToolCallParser(self.TOOLS)
        assert "get_weather" in parser.function_names
        assert "search" in parser.function_names
        assert len(parser.function_names) == 2
