"""
vllm-i64 :: Test API Server

Tests aiohttp endpoints without starting a real server:
  - POST /v1/completions
  - POST /v1/chat/completions
  - GET /health
  - GET /v1/models

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
from vllm_i64.api.server import I64Server


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


@pytest_asyncio.fixture
async def client(server):
    app = server.create_app()
    async with TestClient(TestServer(app)) as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "test-model"
    assert "engine" in data


@pytest.mark.asyncio
async def test_models(client):
    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


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
    assert 1 <= data["choices"][0]["usage"]["completion_tokens"] <= 5


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
