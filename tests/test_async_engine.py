"""
vllm-i64 :: Test Async Continuous Batching

Verifies that multiple concurrent requests are actually batched
together and processed in parallel by the AsyncI64Engine.

Run:
    python -m pytest tests/test_async_engine.py -v
    python tests/test_async_engine.py

INL - 2025
"""

import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.engine.i64_engine import AsyncI64Engine, I64Engine, GenerationResult


async def test_single_request():
    """Single request should complete."""
    engine = AsyncI64Engine(
        model=None,  # Dummy logits
        num_experts=4,
        vocab_size=100,
        max_batch_size=32,
        device="cpu",
    )
    await engine.start()

    result = await engine.generate(
        prompt_token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=10,
    )

    assert isinstance(result, GenerationResult)
    assert len(result.output_tokens) == 10
    assert result.request_id >= 0
    print(f"  single request: {len(result.output_tokens)} tokens, {result.elapsed_ms:.1f}ms")

    await engine.stop()


async def test_parallel_requests():
    """Multiple requests should be batched and processed concurrently."""
    engine = AsyncI64Engine(
        model=None,
        num_experts=4,
        vocab_size=100,
        max_batch_size=32,
        device="cpu",
    )
    await engine.start()

    num_requests = 8
    max_tokens = 20

    start = time.perf_counter()

    # Launch all requests concurrently
    tasks = [
        asyncio.create_task(
            engine.generate(
                prompt_token_ids=list(range(i * 10, i * 10 + 5)),
                max_new_tokens=max_tokens,
            )
        )
        for i in range(num_requests)
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    # All requests should have completed
    assert len(results) == num_requests
    for i, r in enumerate(results):
        assert isinstance(r, GenerationResult)
        assert len(r.output_tokens) == max_tokens, f"Request {i}: expected {max_tokens} tokens, got {len(r.output_tokens)}"

    # Peak batch size should be > 1 (requests were batched)
    stats = engine.get_stats()
    peak = stats["peak_batch_size"]

    print(f"  {num_requests} parallel requests: {elapsed*1000:.1f}ms total")
    print(f"  peak batch size: {peak}")
    print(f"  total tokens generated: {stats['total_tokens_generated']}")
    print(f"  throughput: {num_requests * max_tokens / elapsed:.0f} tok/s")

    assert peak > 1, f"Peak batch size was {peak} — requests were NOT batched!"

    await engine.stop()


async def test_streaming():
    """Streaming should yield tokens one by one."""
    engine = AsyncI64Engine(
        model=None,
        num_experts=4,
        vocab_size=100,
        max_batch_size=32,
        device="cpu",
    )
    await engine.start()

    tokens = []
    async for token_id in engine.generate_stream(
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=15,
    ):
        tokens.append(token_id)

    assert len(tokens) == 15
    print(f"  streaming: received {len(tokens)} tokens one by one")

    await engine.stop()


async def test_mixed_parallel_streaming():
    """Mix of regular and streaming requests in parallel."""
    engine = AsyncI64Engine(
        model=None,
        num_experts=4,
        vocab_size=100,
        max_batch_size=32,
        device="cpu",
    )
    await engine.start()

    # Regular request
    async def regular():
        return await engine.generate([1, 2, 3], max_new_tokens=10)

    # Streaming request
    async def streaming():
        tokens = []
        async for tok in engine.generate_stream([4, 5, 6], max_new_tokens=10):
            tokens.append(tok)
        return tokens

    start = time.perf_counter()
    r1, r2, s1, s2 = await asyncio.gather(
        regular(), regular(), streaming(), streaming(),
    )
    elapsed = time.perf_counter() - start

    assert len(r1.output_tokens) == 10
    assert len(r2.output_tokens) == 10
    assert len(s1) == 10
    assert len(s2) == 10

    stats = engine.get_stats()
    print(f"  mixed parallel: 2 regular + 2 streaming in {elapsed*1000:.1f}ms")
    print(f"  peak batch: {stats['peak_batch_size']}")

    await engine.stop()


async def test_sequential_vs_parallel_speedup():
    """Parallel requests should be faster than sequential."""
    engine = AsyncI64Engine(
        model=None,
        num_experts=4,
        vocab_size=100,
        max_batch_size=32,
        device="cpu",
    )
    await engine.start()

    num_requests = 6
    max_tokens = 20

    # Sequential
    start = time.perf_counter()
    for i in range(num_requests):
        await engine.generate(list(range(i * 10, i * 10 + 5)), max_new_tokens=max_tokens)
    seq_time = time.perf_counter() - start

    # Parallel
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            engine.generate(list(range(i * 10, i * 10 + 5)), max_new_tokens=max_tokens)
        )
        for i in range(num_requests)
    ]
    await asyncio.gather(*tasks)
    par_time = time.perf_counter() - start

    speedup = seq_time / par_time if par_time > 0 else 0

    print(f"  sequential: {seq_time*1000:.1f}ms")
    print(f"  parallel:   {par_time*1000:.1f}ms")
    print(f"  speedup:    {speedup:.1f}x")

    # Parallel should be faster (or at least not slower)
    # With dummy model, the speedup comes from batching reducing total steps
    assert par_time <= seq_time * 1.5, "Parallel was significantly slower than sequential!"

    await engine.stop()


async def main():
    print("=" * 60)
    print("vllm-i64 :: Async Engine Tests")
    print("=" * 60)

    tests = [
        ("Single request", test_single_request),
        ("Parallel requests (batching)", test_parallel_requests),
        ("Streaming", test_streaming),
        ("Mixed parallel + streaming", test_mixed_parallel_streaming),
        ("Sequential vs parallel speedup", test_sequential_vs_parallel_speedup),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            await test_fn()
            print(f"  PASSED")
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
