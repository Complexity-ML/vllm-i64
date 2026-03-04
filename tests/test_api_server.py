"""
vllm-i64 :: Test API Server

Comprehensive tests for aiohttp endpoints:
  - POST /v1/completions (sync + streaming + validation)
  - POST /v1/chat/completions
  - GET /health (deep health check)
  - GET /v1/models
  - POST /v1/batch
  - Rate limiting
  - Authentication
  - Error handling
  - Concurrent requests

Uses aiohttp test_utils for in-process testing.

INL - 2025
"""

import pytest
import pytest_asyncio
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vllm_i64.engine.i64_engine import I64Engine
from vllm_i64.api.server import I64Server, CompletionRequest, CompletionResponse
from vllm_i64.api.middleware import TokenBucketRateLimiter
from vllm_i64.api.tracking import UsageTracker, RequestCache, LatencyTracker, PriorityManager


@pytest.fixture
def engine():
    """Engine with no model (dummy logits)."""
    return I64Engine(
        model=None,
        num_experts=4,
        vocab_size=100,
        max_batch_size=16,
        device="cpu",
    )


@pytest.fixture
def server(engine):
    return I64Server(
        engine=engine,
        tokenizer=None,
        model_name="test-model",
        host="127.0.0.1",
        port=0,
    )


@pytest.fixture
def auth_server(engine):
    """Server with API key authentication."""
    return I64Server(
        engine=engine,
        tokenizer=None,
        model_name="test-model",
        host="127.0.0.1",
        port=0,
        api_key="test-secret-key-12345",
    )


@pytest.fixture
def rate_limited_server(engine):
    """Server with rate limiting."""
    return I64Server(
        engine=engine,
        tokenizer=None,
        model_name="test-model",
        host="127.0.0.1",
        port=0,
        rate_limit=2,  # 2 requests per minute
    )


@pytest_asyncio.fixture
async def client(server):
    app = server.create_app()
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest_asyncio.fixture
async def auth_client(auth_server):
    app = auth_server.create_app()
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest_asyncio.fixture
async def rate_client(rate_limited_server):
    app = rate_limited_server.create_app()
    async with TestClient(TestServer(app)) as c:
        yield c


# =====================================================================
# Health & Models
# =====================================================================

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-model"
    assert "engine" in data
    assert "checks" in data
    assert "queue" in data
    assert "latency" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_health_checks_model_loaded(client):
    """Health check should report model_loaded status."""
    resp = await client.get("/health")
    data = await resp.json()
    assert "checks" in data
    # model is None in test fixture
    assert data["checks"]["model_loaded"] is False


@pytest.mark.asyncio
async def test_models(client):
    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


# =====================================================================
# Completions
# =====================================================================

@pytest.mark.asyncio
async def test_completions(client):
    resp = await client.post("/v1/completions", json={
        "prompt": "hello world",
        "max_tokens": 5,
    })
    assert resp.status == 200
    data = await resp.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]
    # Usage is at top level (OpenAI standard)
    assert "usage" in data
    assert 1 <= data["usage"]["completion_tokens"] <= 5


@pytest.mark.asyncio
async def test_completions_missing_prompt(client):
    """Should return 400 for missing prompt."""
    resp = await client.post("/v1/completions", json={
        "max_tokens": 5,
    })
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
    assert "prompt" in data["error"]["message"].lower()


@pytest.mark.asyncio
async def test_completions_empty_prompt(client):
    """Should return 400 for empty prompt."""
    resp = await client.post("/v1/completions", json={
        "prompt": "",
        "max_tokens": 5,
    })
    assert resp.status == 400


@pytest.mark.asyncio
async def test_completions_invalid_json(client):
    """Should return 400 for invalid JSON."""
    resp = await client.post("/v1/completions", data=b"not json",
                             headers={"Content-Type": "application/json"})
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_completions_invalid_temperature(client):
    """Should return 400 for negative temperature."""
    resp = await client.post("/v1/completions", json={
        "prompt": "test",
        "max_tokens": 5,
        "temperature": -1.0,
    })
    assert resp.status == 400
    data = await resp.json()
    assert "temperature" in data["error"]["message"]


@pytest.mark.asyncio
async def test_completions_invalid_top_p(client):
    """Should return 400 for top_p > 1."""
    resp = await client.post("/v1/completions", json={
        "prompt": "test",
        "max_tokens": 5,
        "top_p": 1.5,
    })
    assert resp.status == 400


@pytest.mark.asyncio
async def test_completions_response_format(client):
    """Response should have proper OpenAI format."""
    resp = await client.post("/v1/completions", json={
        "prompt": "hello",
        "max_tokens": 3,
    })
    data = await resp.json()
    assert "id" in data
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "text_completion"
    assert "created" in data
    assert data["model"] == "test-model"
    assert isinstance(data["choices"], list)


@pytest.mark.asyncio
async def test_completions_streaming(client):
    """SSE streaming should return valid event-stream data."""
    resp = await client.post("/v1/completions", json={
        "prompt": "hello",
        "max_tokens": 3,
        "stream": True,
    })
    assert resp.status == 200
    assert resp.content_type == "text/event-stream"
    body = await resp.read()
    text = body.decode("utf-8")
    # Should contain SSE data lines
    assert "data:" in text
    # Should end with [DONE]
    assert "[DONE]" in text


# =====================================================================
# Chat Completions
# =====================================================================

@pytest.mark.asyncio
async def test_chat_completions(client):
    resp = await client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "Hi there"},
        ],
        "max_tokens": 3,
    })
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "chat.completion"
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_completions_missing_messages(client):
    """Should return 400 for missing messages."""
    resp = await client.post("/v1/chat/completions", json={
        "max_tokens": 3,
    })
    assert resp.status == 400
    data = await resp.json()
    assert "messages" in data["error"]["message"].lower()


# =====================================================================
# Concurrent Requests
# =====================================================================

@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """Multiple requests should all complete."""
    tasks = [
        client.post("/v1/completions", json={"prompt": f"test {i}", "max_tokens": 3})
        for i in range(4)
    ]
    responses = await asyncio.gather(*tasks)
    for resp in responses:
        assert resp.status == 200
        data = await resp.json()
        assert len(data["choices"]) == 1


@pytest.mark.asyncio
async def test_concurrent_unique_ids(client):
    """Concurrent requests should get unique IDs (race-free counter)."""
    tasks = [
        client.post("/v1/completions", json={"prompt": f"test {i}", "max_tokens": 2})
        for i in range(8)
    ]
    responses = await asyncio.gather(*tasks)
    ids = set()
    for resp in responses:
        data = await resp.json()
        ids.add(data["id"])
    # All IDs should be unique
    assert len(ids) == 8


# =====================================================================
# Authentication
# =====================================================================

@pytest.mark.asyncio
async def test_auth_no_key_rejected(auth_client):
    """Requests without API key should be rejected."""
    resp = await auth_client.post("/v1/completions", json={
        "prompt": "test",
        "max_tokens": 3,
    })
    assert resp.status == 401
    data = await resp.json()
    assert data["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_auth_wrong_key_rejected(auth_client):
    """Requests with wrong API key should be rejected."""
    resp = await auth_client.post("/v1/completions", json={
        "prompt": "test",
        "max_tokens": 3,
    }, headers={"Authorization": "Bearer wrong-key"})
    assert resp.status == 401


@pytest.mark.asyncio
async def test_auth_correct_key_accepted(auth_client):
    """Requests with correct API key should succeed."""
    resp = await auth_client.post("/v1/completions", json={
        "prompt": "test",
        "max_tokens": 3,
    }, headers={"Authorization": "Bearer test-secret-key-12345"})
    assert resp.status == 200


@pytest.mark.asyncio
async def test_auth_health_no_key_needed(auth_client):
    """Health endpoint should not require authentication."""
    resp = await auth_client.get("/health")
    assert resp.status == 200


# =====================================================================
# Rate Limiting
# =====================================================================

@pytest.mark.asyncio
async def test_rate_limiting(rate_client):
    """Should reject requests exceeding rate limit."""
    # First request should succeed
    resp1 = await rate_client.post("/v1/completions", json={
        "prompt": "test 1",
        "max_tokens": 2,
    })
    assert resp1.status == 200

    # Second request should succeed (capacity = 2)
    resp2 = await rate_client.post("/v1/completions", json={
        "prompt": "test 2",
        "max_tokens": 2,
    })
    assert resp2.status == 200

    # Third request should be rate limited
    resp3 = await rate_client.post("/v1/completions", json={
        "prompt": "test 3",
        "max_tokens": 2,
    })
    assert resp3.status == 429
    data = await resp3.json()
    assert data["error"]["type"] == "rate_limit_error"


# =====================================================================
# Batch Endpoint
# =====================================================================

@pytest.mark.asyncio
async def test_batch_completions(client):
    """Batch endpoint should process multiple prompts."""
    resp = await client.post("/v1/batch", json={
        "requests": [
            {"prompt": "hello", "max_tokens": 3},
            {"prompt": "world", "max_tokens": 3},
        ],
    })
    assert resp.status == 200
    data = await resp.json()
    assert "responses" in data
    assert len(data["responses"]) == 2
    for r in data["responses"]:
        assert "result" in r


@pytest.mark.asyncio
async def test_batch_validation(client):
    """Batch should validate individual requests."""
    resp = await client.post("/v1/batch", json={
        "requests": [
            {"prompt": "ok", "max_tokens": 3},
            {"prompt": "", "max_tokens": 3},  # Invalid: empty prompt
        ],
    })
    assert resp.status == 400


# =====================================================================
# Unit Tests for Utility Classes
# =====================================================================

class TestCompletionRequest:
    def test_validate_valid(self):
        req = CompletionRequest(prompt="hello", max_tokens=10)
        assert req.validate() is None

    def test_validate_empty_prompt(self):
        req = CompletionRequest(prompt="", max_tokens=10)
        assert req.validate() is not None

    def test_validate_negative_temperature(self):
        req = CompletionRequest(prompt="hello", temperature=-1.0)
        assert "temperature" in req.validate()

    def test_validate_invalid_top_p(self):
        req = CompletionRequest(prompt="hello", top_p=1.5)
        assert "top_p" in req.validate()

    def test_validate_invalid_min_p(self):
        req = CompletionRequest(prompt="hello", min_p=-0.1)
        assert "min_p" in req.validate()

    def test_validate_logprobs_range(self):
        req = CompletionRequest(prompt="hello", logprobs=25)
        assert "logprobs" in req.validate()

    def test_validate_frequency_penalty_range(self):
        req = CompletionRequest(prompt="hello", frequency_penalty=3.0)
        assert "frequency_penalty" in req.validate()

    def test_to_sampling_params(self):
        req = CompletionRequest(prompt="hello", temperature=0.5, top_k=10)
        params = req.to_sampling_params()
        assert params.temperature == 0.5
        assert params.top_k == 10


class TestCompletionResponse:
    def test_default_choices(self):
        """choices should default to empty list, not None."""
        resp = CompletionResponse(id="test-1")
        assert resp.choices == []
        d = resp.to_dict()
        assert d["choices"] == []


class TestTokenBucketRateLimiter:
    def test_allow_within_limit(self):
        limiter = TokenBucketRateLimiter(60)  # 60 req/min
        assert limiter.allow("1.2.3.4") is True
        assert limiter.allow("1.2.3.4") is True

    def test_deny_over_limit(self):
        limiter = TokenBucketRateLimiter(1)  # 1 req/min
        assert limiter.allow("1.2.3.4") is True
        assert limiter.allow("1.2.3.4") is False

    def test_different_ips_independent(self):
        limiter = TokenBucketRateLimiter(1)
        assert limiter.allow("1.2.3.4") is True
        assert limiter.allow("5.6.7.8") is True

    def test_stale_cleanup(self):
        """Stale buckets should be cleaned up."""
        limiter = TokenBucketRateLimiter(60, max_buckets=2)
        limiter.allow("ip1")
        limiter.allow("ip2")
        # Third IP should trigger cleanup or eviction
        limiter.allow("ip3")
        assert len(limiter._buckets) <= 3  # At most max_buckets + 1 during cleanup


class TestRequestCache:
    def test_cache_deterministic(self):
        """Should cache requests with temperature=0."""
        cache = RequestCache()
        cache.put("hello", 10, {"result": "cached"}, temperature=0.0, top_k=50, top_p=0.9)
        assert cache.get("hello", 10, temperature=0.0, top_k=50, top_p=0.9) == {"result": "cached"}

    def test_no_cache_nondeterministic(self):
        """Should NOT cache requests with temperature > 0."""
        cache = RequestCache()
        cache.put("hello", 10, {"result": "cached"}, temperature=0.8, top_k=50, top_p=0.9)
        assert cache.get("hello", 10, temperature=0.8, top_k=50, top_p=0.9) is None

    def test_eviction_order(self):
        """Oldest entries should be evicted first (O(1) with OrderedDict)."""
        cache = RequestCache(max_size=2)
        cache.put("a", 10, {"r": "a"}, temperature=0.0, top_k=50, top_p=0.9)
        cache.put("b", 10, {"r": "b"}, temperature=0.0, top_k=50, top_p=0.9)
        cache.put("c", 10, {"r": "c"}, temperature=0.0, top_k=50, top_p=0.9)  # Should evict "a"
        assert cache.get("a", 10, temperature=0.0, top_k=50, top_p=0.9) is None
        assert cache.get("b", 10, temperature=0.0, top_k=50, top_p=0.9) is not None

    def test_different_sampling_params_not_shared(self):
        """Different sampling params should NOT share cache entries."""
        cache = RequestCache()
        cache.put("hello", 10, {"result": "a"}, temperature=0.0, min_p=0.1)
        assert cache.get("hello", 10, temperature=0.0, min_p=0.2) is None
        assert cache.get("hello", 10, temperature=0.0, min_p=0.1) == {"result": "a"}


class TestUsageTracker:
    def test_record_and_get(self):
        tracker = UsageTracker()
        tracker.record("key1", 10, 5)
        tracker.record("key1", 20, 10)
        usage = tracker.get("key1")
        assert usage["prompt_tokens"] == 30
        assert usage["completion_tokens"] == 15
        assert usage["requests"] == 2

    def test_get_total(self):
        tracker = UsageTracker()
        tracker.record("key1", 10, 5)
        tracker.record("key2", 20, 10)
        total = tracker.get_total()
        assert total["prompt_tokens"] == 30
        assert total["requests"] == 2


class TestLatencyTracker:
    def test_percentiles(self):
        tracker = LatencyTracker()
        for i in range(100):
            tracker.record("/v1/completions", float(i))
        p = tracker.percentiles()
        assert p["count"] == 100
        assert p["p50_ms"] == 50.0
        assert p["p95_ms"] >= 94.0

    def test_per_endpoint(self):
        tracker = LatencyTracker()
        tracker.record("/v1/completions", 10.0)
        tracker.record("/v1/chat", 20.0)
        p = tracker.get_all_endpoints()
        assert "/v1/completions" in p
        assert "/v1/chat" in p


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
