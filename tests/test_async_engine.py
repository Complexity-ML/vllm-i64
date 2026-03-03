"""
vllm-i64 :: Async Engine Tests

Tests for AsyncI64Engine lifecycle, generation, streaming,
concurrency, stats, and graceful shutdown.

Run:
    python -m pytest tests/test_async_engine.py -v
    python tests/test_async_engine.py

Note: project uses pytest-asyncio; asyncio_mode = "auto" can be set
in pyproject.toml [tool.pytest.ini_options] if desired.

INL - 2025
"""

import pytest
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine, GenerationResult
from vllm_i64.core.sampling import SamplingParams


@pytest.mark.asyncio
async def test_start_stop():
    """Engine start/stop lifecycle."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    assert engine._running is True
    await engine.stop()
    assert engine._running is False


@pytest.mark.asyncio
async def test_generate_basic():
    """Basic async generation."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    result = await engine.generate(prompt_token_ids=[1, 2, 3], max_new_tokens=5)
    assert isinstance(result, GenerationResult)
    assert len(result.output_tokens) == 5
    assert result.finish_reason == "length"
    await engine.stop()


@pytest.mark.asyncio
async def test_generate_with_sampling_params():
    """Generation with explicit SamplingParams."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    params = SamplingParams(temperature=0.0, top_k=1, max_tokens=3)
    result = await engine.generate([10, 20], max_new_tokens=3, sampling_params=params)
    assert len(result.output_tokens) == 3
    await engine.stop()


@pytest.mark.asyncio
async def test_concurrent_generates():
    """Multiple concurrent generates should all complete."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    tasks = [engine.generate([i + 1], max_new_tokens=3) for i in range(5)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    for r in results:
        assert len(r.output_tokens) == 3
    await engine.stop()


@pytest.mark.asyncio
async def test_generate_stream():
    """Streaming should yield individual tokens."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    tokens = []
    async for token_id in engine.generate_stream([1, 2], max_new_tokens=4):
        tokens.append(token_id)
    assert len(tokens) == 4
    await engine.stop()


@pytest.mark.asyncio
async def test_from_sync_engine():
    """AsyncI64Engine.from_sync_engine wraps an existing I64Engine."""
    sync = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    engine = AsyncI64Engine.from_sync_engine(sync)
    await engine.start()
    result = await engine.generate([1], max_new_tokens=2)
    assert len(result.output_tokens) == 2
    await engine.stop()


@pytest.mark.asyncio
async def test_stats():
    """get_stats returns expected keys after generation."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    await engine.generate([1, 2], max_new_tokens=3)
    stats = engine.get_stats()
    assert "active_requests" in stats
    assert "peak_batch_size" in stats
    assert stats["total_tokens_generated"] > 0
    await engine.stop()


@pytest.mark.asyncio
async def test_drain_on_stop():
    """Stop should drain active requests before shutting down."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    # Start a generate in background
    task = asyncio.create_task(engine.generate([1], max_new_tokens=3))
    await asyncio.sleep(0.05)  # Let it start processing
    await engine.stop(drain_timeout=5.0)
    # Task should complete (not be cancelled)
    result = await task
    assert isinstance(result, GenerationResult)


@pytest.mark.asyncio
async def test_active_requests_counter():
    """active_requests should be 0 after all requests complete."""
    engine = AsyncI64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
    await engine.start()
    assert engine.active_requests == 0
    await engine.generate([1], max_new_tokens=2)
    assert engine.active_requests == 0  # Should be 0 after completion
    await engine.stop()


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("vllm-i64 :: Async Engine Tests")
    print("=" * 60)

    tests = [
        ("start / stop lifecycle", test_start_stop),
        ("basic generation", test_generate_basic),
        ("generation with SamplingParams", test_generate_with_sampling_params),
        ("concurrent generates", test_concurrent_generates),
        ("streaming generation", test_generate_stream),
        ("from_sync_engine", test_from_sync_engine),
        ("stats after generation", test_stats),
        ("drain on stop", test_drain_on_stop),
        ("active_requests counter", test_active_requests_counter),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            await test_fn()
            print("  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
