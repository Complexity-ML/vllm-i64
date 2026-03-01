"""
vllm-i64 :: Test Features V2

Tests for the second batch of features:
  - Sliding window attention
  - Load shedding (max pending, 503)
  - Request deduplication (RequestCache)
  - Embeddings (engine.embed)
  - Usage tracking (UsageTracker)
  - LoRA adapter serving (engine integration)
  - Detailed health check

Run:
    python -m pytest tests/test_features_v2.py -v

INL - 2025
"""

import pytest
import torch
import time
import math
import sys
import os
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 1. Sliding Window Attention
# =====================================================================

from vllm_i64.layers.attention import naive_varlen_attention, naive_cached_attention


class TestSlidingWindowAttention:
    """Test sliding window attention masking."""

    def test_no_window_baseline(self):
        """Without sliding window, all positions attend to all prior positions."""
        torch.manual_seed(42)
        n = 8
        q = torch.randn(n, 2, 16)
        k = torch.randn(n, 2, 16)
        v = torch.randn(n, 2, 16)
        out = naive_varlen_attention(q, k, v, [n], num_kv_groups=1)
        assert out.shape == (n, 2, 16)

    def test_sliding_window_limits_attention(self):
        """With sliding window, output changes vs full attention."""
        torch.manual_seed(42)
        n = 16
        q = torch.randn(n, 2, 16)
        k = torch.randn(n, 2, 16)
        v = torch.randn(n, 2, 16)

        out_full = naive_varlen_attention(q, k, v, [n], num_kv_groups=1)
        out_sw = naive_varlen_attention(q, k, v, [n], num_kv_groups=1, sliding_window=4)

        # First 4 tokens should be identical (window covers all prior)
        assert torch.allclose(out_full[:4], out_sw[:4], atol=1e-5)
        # Later tokens should differ (window excludes some)
        assert not torch.allclose(out_full[-1:], out_sw[-1:], atol=1e-3)

    def test_sliding_window_1(self):
        """Window=1 means each token only attends to itself."""
        torch.manual_seed(42)
        n = 8
        q = torch.randn(n, 1, 16)
        k = torch.randn(n, 1, 16)
        v = torch.randn(n, 1, 16)

        out_sw1 = naive_varlen_attention(q, k, v, [n], num_kv_groups=1, sliding_window=1)
        # With window=1, each output should only depend on its own V
        # Verify shape
        assert out_sw1.shape == (n, 1, 16)

    def test_sliding_window_cached(self):
        """Sliding window in cached attention."""
        torch.manual_seed(42)
        n_q = 2
        n_kv = 10
        q = torch.randn(n_q, 2, 16)
        k = torch.randn(n_kv, 2, 16)
        v = torch.randn(n_kv, 2, 16)
        positions = torch.tensor([8, 9], dtype=torch.int32)

        out_full = naive_cached_attention(q, k, v, 1, positions)
        out_sw = naive_cached_attention(q, k, v, 1, positions, sliding_window=4)

        assert out_full.shape == (n_q, 2, 16)
        assert out_sw.shape == (n_q, 2, 16)
        # Results should differ since window restricts attention
        assert not torch.allclose(out_full, out_sw, atol=1e-3)

    def test_sliding_window_large_window_equals_full(self):
        """Window larger than sequence = same as full attention."""
        torch.manual_seed(42)
        n = 6
        q = torch.randn(n, 2, 16)
        k = torch.randn(n, 2, 16)
        v = torch.randn(n, 2, 16)

        out_full = naive_varlen_attention(q, k, v, [n], num_kv_groups=1)
        out_sw = naive_varlen_attention(q, k, v, [n], num_kv_groups=1, sliding_window=100)

        assert torch.allclose(out_full, out_sw, atol=1e-5)

    def test_sliding_window_multi_seq(self):
        """Sliding window with multiple sequences."""
        torch.manual_seed(42)
        q = torch.randn(12, 2, 16)  # 2 seqs: 4+8
        k = torch.randn(12, 2, 16)
        v = torch.randn(12, 2, 16)

        out = naive_varlen_attention(q, k, v, [4, 8], num_kv_groups=1, sliding_window=3)
        assert out.shape == (12, 2, 16)
        assert torch.isfinite(out).all()


# =====================================================================
# 2. Load Shedding
# =====================================================================

from vllm_i64.api.server import I64Server, TokenBucketRateLimiter


class TestLoadShedding:
    """Test load shedding (max_pending parameter)."""

    def test_max_pending_stored(self):
        """max_pending is stored on server."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine, max_pending=50)
        assert server._max_pending == 50

    def test_max_pending_zero_means_unlimited(self):
        """max_pending=0 means no load shedding."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine, max_pending=0)
        assert server._max_pending == 0

    def test_load_shed_middleware_registered(self):
        """Load shed middleware is registered when max_pending > 0."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine, max_pending=10)
        app = server.create_app()
        # Middlewares should include load_shed
        assert len(app.middlewares) >= 2  # cors + load_shed at minimum


# =====================================================================
# 3. Request Deduplication
# =====================================================================

from vllm_i64.api.server import RequestCache


class TestRequestCache:
    """Test request deduplication cache."""

    def test_cache_miss(self):
        """Empty cache returns None."""
        cache = RequestCache()
        assert cache.get("hello", 0.0, 50, 0.9, 256) is None

    def test_cache_hit_deterministic(self):
        """Deterministic requests (temp=0) are cached."""
        cache = RequestCache()
        result = {"text": "output"}
        cache.put("hello", 0.0, 50, 0.9, 256, result)
        assert cache.get("hello", 0.0, 50, 0.9, 256) == result

    def test_cache_skip_nondeterministic(self):
        """Non-deterministic requests (temp>0) are NOT cached."""
        cache = RequestCache()
        result = {"text": "output"}
        cache.put("hello", 0.8, 50, 0.9, 256, result)
        assert cache.get("hello", 0.8, 50, 0.9, 256) is None

    def test_cache_different_params(self):
        """Different params produce different cache keys."""
        cache = RequestCache()
        cache.put("hello", 0.0, 50, 0.9, 256, {"a": 1})
        cache.put("hello", 0.0, 50, 0.9, 128, {"a": 2})
        assert cache.get("hello", 0.0, 50, 0.9, 256) == {"a": 1}
        assert cache.get("hello", 0.0, 50, 0.9, 128) == {"a": 2}

    def test_cache_expiration(self):
        """Expired entries return None."""
        cache = RequestCache(ttl_seconds=0.01)
        cache.put("hello", 0.0, 50, 0.9, 256, {"text": "cached"})
        assert cache.get("hello", 0.0, 50, 0.9, 256) is not None
        time.sleep(0.02)
        assert cache.get("hello", 0.0, 50, 0.9, 256) is None

    def test_cache_eviction(self):
        """Cache evicts oldest entry when full."""
        cache = RequestCache(max_size=2)
        cache.put("a", 0.0, 50, 0.9, 256, {"a": 1})
        cache.put("b", 0.0, 50, 0.9, 256, {"b": 2})
        assert cache.size == 2
        cache.put("c", 0.0, 50, 0.9, 256, {"c": 3})
        assert cache.size == 2  # Still 2, oldest evicted

    def test_cache_hit_rate_info(self):
        """hit_rate_info returns cache stats."""
        cache = RequestCache(max_size=100)
        cache.put("x", 0.0, 50, 0.9, 256, {"x": 1})
        info = cache.hit_rate_info
        assert info["cached_entries"] == 1
        assert info["max_size"] == 100


# =====================================================================
# 4. Embeddings
# =====================================================================

class TestEmbeddings:
    """Test engine.embed() method."""

    def test_embed_with_model(self):
        """embed() returns embedding vector when model has embed_tokens."""
        import torch.nn as nn

        class MockConfig:
            num_experts = 4
            vocab_size = 100
            hidden_size = 32
            num_hidden_layers = 2
            num_key_value_heads = 2
            num_attention_heads = 4
            head_dim = 8
            eos_token_id = 0

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.embed_tokens = nn.Embedding(100, 32)
                self.norm = None  # No norm for simplicity

            def forward(self, **kwargs):
                return torch.randn(1, 100)

        model = MockModel()
        model.eval()

        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=model, num_experts=4, vocab_size=100, device="cpu")
        embedding = engine.embed([1, 2, 3, 4])

        assert isinstance(embedding, list)
        assert len(embedding) == 32  # hidden_size
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_no_model_raises(self):
        """embed() raises when no model loaded."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        with pytest.raises(RuntimeError):
            engine.embed([1, 2, 3])

    def test_embed_deterministic(self):
        """Same input produces same embedding."""
        import torch.nn as nn

        class MockConfig:
            num_experts = 4
            vocab_size = 100
            hidden_size = 16
            num_hidden_layers = 1
            num_key_value_heads = 1
            num_attention_heads = 1
            head_dim = 16
            eos_token_id = 0

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.embed_tokens = nn.Embedding(100, 16)
                self.norm = None

            def forward(self, **kwargs):
                return torch.randn(1, 100)

        model = MockModel()
        model.eval()

        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=model, num_experts=4, vocab_size=100, device="cpu")
        e1 = engine.embed([10, 20, 30])
        e2 = engine.embed([10, 20, 30])
        assert e1 == e2


# =====================================================================
# 5. Usage Tracking
# =====================================================================

from vllm_i64.api.server import UsageTracker


class TestUsageTracker:
    """Test per-API-key usage tracking."""

    def test_empty_tracker(self):
        """New tracker has zero usage."""
        tracker = UsageTracker()
        total = tracker.get_total()
        assert total["prompt_tokens"] == 0
        assert total["completion_tokens"] == 0
        assert total["requests"] == 0

    def test_record_usage(self):
        """Recording usage accumulates correctly."""
        tracker = UsageTracker()
        tracker.record("key-1", prompt_tokens=100, completion_tokens=50)
        tracker.record("key-1", prompt_tokens=200, completion_tokens=100)

        usage = tracker.get("key-1")
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 150
        assert usage["requests"] == 2

    def test_per_key_isolation(self):
        """Different API keys have separate counters."""
        tracker = UsageTracker()
        tracker.record("key-a", 100, 50)
        tracker.record("key-b", 200, 100)

        assert tracker.get("key-a")["prompt_tokens"] == 100
        assert tracker.get("key-b")["prompt_tokens"] == 200

    def test_total_aggregation(self):
        """get_total() sums across all keys."""
        tracker = UsageTracker()
        tracker.record("key-a", 100, 50)
        tracker.record("key-b", 200, 100)

        total = tracker.get_total()
        assert total["prompt_tokens"] == 300
        assert total["completion_tokens"] == 150
        assert total["requests"] == 2

    def test_unknown_key_returns_zero(self):
        """Querying unknown key returns zeroes."""
        tracker = UsageTracker()
        usage = tracker.get("nonexistent")
        assert usage["prompt_tokens"] == 0
        assert usage["requests"] == 0


# =====================================================================
# 6. LoRA Adapter Serving
# =====================================================================

from vllm_i64.layers.lora import LoRALinear, LoRAManager


class TestLoRAServing:
    """Test LoRA adapter integration with engine."""

    def test_lora_manager_auto_wrap(self):
        """LoRAManager.auto_wrap wraps matching linear layers."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Module()
                self.layer.q_proj = nn.Linear(32, 32, bias=False)
                self.layer.k_proj = nn.Linear(32, 32, bias=False)
                self.layer.v_proj = nn.Linear(32, 32, bias=False)
                self.layer.o_proj = nn.Linear(32, 32, bias=False)
                self.layer.mlp = nn.Linear(32, 32, bias=False)

        model = SimpleModel()
        mgr = LoRAManager(model)
        mgr.auto_wrap()

        # Should wrap q/k/v/o but NOT mlp
        assert len(mgr._lora_modules) == 4
        assert isinstance(model.layer.q_proj, LoRALinear)
        assert isinstance(model.layer.mlp, nn.Linear)  # NOT wrapped

    def test_lora_load_unload(self):
        """Load and unload LoRA adapters."""
        import torch.nn as nn

        base = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(base)

        A = torch.randn(16, 4)
        B = torch.randn(4, 16)
        lora.load_adapter(1, A, B, scaling=1.0)

        assert lora.num_adapters == 1
        assert 1 in lora.adapter_ids

        lora.unload_adapter(1)
        assert lora.num_adapters == 0

    def test_lora_forward_with_adapter(self):
        """LoRA forward modifies output when adapter is active."""
        import torch.nn as nn

        torch.manual_seed(42)
        base = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(base)

        x = torch.randn(4, 16)
        out_base = lora(x)

        A = torch.randn(16, 4) * 0.1
        B = torch.randn(4, 16) * 0.1
        lora.load_adapter(1, A, B, scaling=1.0)
        lora.set_active_adapter(1)
        out_lora = lora(x)

        # Output should differ
        assert not torch.allclose(out_base, out_lora, atol=1e-5)

    def test_lora_forward_without_adapter(self):
        """LoRA forward matches base when no adapter active."""
        import torch.nn as nn

        torch.manual_seed(42)
        base = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(base)

        x = torch.randn(4, 16)
        out_base = base(x)
        out_lora = lora(x)

        assert torch.allclose(out_base, out_lora)

    def test_engine_lora_methods(self):
        """Engine has LoRA management methods."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")

        # No model → can't enable
        with pytest.raises(RuntimeError):
            engine.enable_lora()

        # list_lora returns empty
        assert engine.list_lora_adapters() == {}

    def test_lora_max_adapters(self):
        """LoRALinear enforces max_adapters limit."""
        import torch.nn as nn

        base = nn.Linear(8, 8, bias=False)
        lora = LoRALinear(base, max_adapters=2)

        A, B = torch.randn(8, 2), torch.randn(2, 8)
        lora.load_adapter(1, A, B)
        lora.load_adapter(2, A, B)

        with pytest.raises(RuntimeError, match="Max adapters"):
            lora.load_adapter(3, A, B)


# =====================================================================
# 7. Detailed Health Check
# =====================================================================

class TestDetailedHealth:
    """Test detailed health check response structure."""

    def test_health_has_uptime(self):
        """Server tracks start time for uptime."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine)
        assert hasattr(server, '_start_time')
        assert server._start_time > 0

    def test_health_has_usage_tracker(self):
        """Server has usage tracker."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine)
        assert isinstance(server._usage_tracker, UsageTracker)

    def test_health_has_request_cache(self):
        """Server has request cache."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine)
        assert isinstance(server._request_cache, RequestCache)

    def test_server_routes_include_new_endpoints(self):
        """create_app registers all new endpoints."""
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        server = I64Server(engine=engine)
        app = server.create_app()

        # Collect all registered routes
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, 'resource')]
        assert "/v1/embeddings" in routes
        assert "/v1/usage" in routes
        assert "/v1/lora/load" in routes
        assert "/v1/lora/unload" in routes
        assert "/v1/lora/list" in routes


# =====================================================================
# 8. Integration: Server with all features
# =====================================================================

class TestServerIntegration:
    """Integration tests for server with all new features enabled."""

    def _make_server(self, **kwargs):
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        return I64Server(engine=engine, **kwargs)

    def test_all_features_init(self):
        """Server initializes with all features enabled."""
        server = self._make_server(
            api_key="test-key",
            rate_limit=60,
            max_pending=100,
        )
        assert server.api_key == "test-key"
        assert server._rate_limiter is not None
        assert server._max_pending == 100
        assert server._usage_tracker is not None
        assert server._request_cache is not None

    def test_app_has_all_middlewares(self):
        """App has all middlewares when features enabled."""
        server = self._make_server(
            api_key="test-key",
            rate_limit=60,
            max_pending=100,
        )
        app = server.create_app()
        # cors + auth + rate_limit + load_shed = 4
        assert len(app.middlewares) == 4

    def test_app_minimal_middlewares(self):
        """App has only cors when no features enabled."""
        server = self._make_server()
        app = server.create_app()
        assert len(app.middlewares) == 1  # just cors
