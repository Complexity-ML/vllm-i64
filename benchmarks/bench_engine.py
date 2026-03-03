"""
vllm-i64 :: Engine Benchmark

Comprehensive benchmarking tool measuring:
  - TTFT  (Time to First Token) — ms
  - ITL   (Inter-Token Latency) — ms/tok
  - Throughput — tok/s (prefill + decode)
  - KV cache efficiency
  - Scheduler overhead

Runs sync and async modes, multiple prompt lengths and batch sizes.

INL - 2025
"""

import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchResult:
    """Single benchmark result."""
    label: str
    num_requests: int
    prompt_len: int
    output_len: int
    total_time_s: float
    ttft_ms: float
    avg_itl_ms: float
    p50_itl_ms: float
    p99_itl_ms: float
    throughput_tok_s: float
    total_tokens: int
    peak_batch: int = 0


def bench_sync(
    num_requests: int = 20,
    prompt_len: int = 64,
    output_len: int = 64,
    num_experts: int = 4,
    vocab_size: int = 32000,
) -> BenchResult:
    """Benchmark sync engine: TTFT, ITL, throughput."""
    from vllm_i64.engine.i64_engine import I64Engine
    from vllm_i64.core.sampling import SamplingParams

    max_seq = prompt_len + output_len + 32
    engine = I64Engine(
        num_experts=num_experts,
        vocab_size=vocab_size,
        max_batch_size=32,
        max_seq_len=max_seq,
        device="cpu",
    )
    engine.sampling_params = SamplingParams(temperature=0.0)

    # Warmup
    for _ in range(3):
        p = list(range(1, min(prompt_len, 32) + 1))
        engine.add_request(p, max_new_tokens=8)
        while engine.scheduler.pending or engine.scheduler.running:
            engine.step()
        engine.scheduler.finished.clear()

    # Benchmark
    ttft_list = []
    itl_list = []
    total_output = 0

    overall_start = time.perf_counter()

    for _ in range(num_requests):
        prompt = list(range(1, prompt_len + 1))
        req_start = time.perf_counter()
        rid = engine.add_request(prompt, max_new_tokens=output_len)

        first_token_time = None
        prev_time = req_start
        tokens = 0

        while True:
            results = engine.step()
            if rid in results:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    ttft_list.append((now - req_start) * 1000)
                else:
                    itl_list.append((now - prev_time) * 1000)
                prev_time = now
                tokens += 1

            finished = any(r.request_id == rid for r in engine.scheduler.finished)
            if finished:
                break

        total_output += tokens
        engine.scheduler.finished = [
            r for r in engine.scheduler.finished if r.request_id != rid
        ]

    total_time = time.perf_counter() - overall_start

    itl_arr = np.array(itl_list) if itl_list else np.array([0.0])

    return BenchResult(
        label="sync",
        num_requests=num_requests,
        prompt_len=prompt_len,
        output_len=output_len,
        total_time_s=round(total_time, 3),
        ttft_ms=round(np.mean(ttft_list), 2) if ttft_list else 0,
        avg_itl_ms=round(float(np.mean(itl_arr)), 2),
        p50_itl_ms=round(float(np.percentile(itl_arr, 50)), 2),
        p99_itl_ms=round(float(np.percentile(itl_arr, 99)), 2),
        throughput_tok_s=round(total_output / total_time, 1),
        total_tokens=total_output,
    )


async def bench_async(
    num_requests: int = 50,
    prompt_len: int = 64,
    output_len: int = 64,
    concurrency: int = 16,
    num_experts: int = 4,
    vocab_size: int = 32000,
) -> BenchResult:
    """Benchmark async engine: concurrent requests, batching throughput."""
    from vllm_i64.engine.i64_engine import AsyncI64Engine
    from vllm_i64.core.sampling import SamplingParams

    max_seq = prompt_len + output_len + 32
    engine = AsyncI64Engine(
        num_experts=num_experts,
        vocab_size=vocab_size,
        max_batch_size=concurrency,
        max_seq_len=max_seq,
        device="cpu",
    )
    engine.engine.sampling_params = SamplingParams(temperature=0.0)
    await engine.start()

    # Warmup
    warmups = []
    for _ in range(3):
        p = list(range(1, min(prompt_len, 32) + 1))
        warmups.append(engine.generate(p, max_new_tokens=8))
    await asyncio.gather(*warmups)
    engine.engine.scheduler.finished.clear()

    # Benchmark concurrent requests
    total_output = 0
    latencies = []  # ms per request
    sem = asyncio.Semaphore(concurrency)

    async def _req():
        nonlocal total_output
        async with sem:
            prompt = list(range(1, prompt_len + 1))
            result = await engine.generate(prompt, max_new_tokens=output_len)
            n = len(result.output_tokens)
            total_output += n
            latencies.append(result.elapsed_ms)

    start = time.perf_counter()
    tasks = [_req() for _ in range(num_requests)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    peak = engine.peak_batch_size
    await engine.stop()

    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    avg_tokens = total_output / max(1, num_requests)
    avg_itl = float(np.mean(lat_arr)) / max(1, avg_tokens)

    return BenchResult(
        label=f"async (c={concurrency})",
        num_requests=num_requests,
        prompt_len=prompt_len,
        output_len=output_len,
        total_time_s=round(total_time, 3),
        ttft_ms=round(float(np.mean(lat_arr)) / max(1, avg_tokens), 2),
        avg_itl_ms=round(avg_itl, 2),
        p50_itl_ms=round(float(np.percentile(lat_arr, 50)) / max(1, avg_tokens), 2),
        p99_itl_ms=round(float(np.percentile(lat_arr, 99)) / max(1, avg_tokens), 2),
        throughput_tok_s=round(total_output / total_time, 1),
        total_tokens=total_output,
        peak_batch=peak,
    )


def bench_kv_cache(
    num_seqs: int = 32,
    seq_len: int = 256,
    num_layers: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    block_size: int = 16,
) -> dict:
    """Benchmark KV cache allocation, write, read, free cycles."""
    import torch
    from vllm_i64.core.kv_cache import PagedKVCache

    num_blocks = max(256, (num_seqs * seq_len) // block_size + 64)
    cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        max_seqs=num_seqs + 1,
        dtype=torch.float32,
        device="cpu",
    )

    # Allocate + write
    start = time.perf_counter()
    for sid in range(num_seqs):
        blocks_needed = (seq_len + block_size - 1) // block_size
        cache.allocate_blocks(sid, blocks_needed)
        for layer in range(num_layers):
            positions = torch.arange(seq_len, dtype=torch.int32)
            k = torch.randn(seq_len, num_kv_heads, head_dim)
            v = torch.randn(seq_len, num_kv_heads, head_dim)
            cache.write_kv_batch(layer, sid, positions, k, v)
    alloc_write_ms = (time.perf_counter() - start) * 1000

    # Read
    start = time.perf_counter()
    for sid in range(num_seqs):
        for layer in range(num_layers):
            cache.read_kv(layer, sid)
    read_ms = (time.perf_counter() - start) * 1000

    # Free (lazy zeroing)
    start = time.perf_counter()
    for sid in range(num_seqs):
        cache.free_sequence(sid)
    free_ms = (time.perf_counter() - start) * 1000

    return {
        "num_seqs": num_seqs,
        "seq_len": seq_len,
        "alloc_write_ms": round(alloc_write_ms, 2),
        "read_ms": round(read_ms, 2),
        "free_ms": round(free_ms, 2),
        "dirty_blocks": len(cache._dirty_blocks),
        "free_blocks": cache.num_free_blocks,
    }


def print_results(results: List[BenchResult]):
    """Pretty-print benchmark table."""
    print()
    header = (
        f"  {'Mode':<20} {'Reqs':>5} {'Prompt':>6} {'Output':>6} "
        f"{'TTFT':>7} {'ITL':>7} {'P50':>7} {'P99':>7} "
        f"{'Tok/s':>9} {'Time':>7} {'Batch':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        row = (
            f"  {r.label:<20} {r.num_requests:>5} {r.prompt_len:>6} {r.output_len:>6} "
            f"{r.ttft_ms:>6.1f}m {r.avg_itl_ms:>6.1f}m {r.p50_itl_ms:>6.1f}m {r.p99_itl_ms:>6.1f}m "
            f"{r.throughput_tok_s:>9.0f} {r.total_time_s:>6.2f}s "
            f"{r.peak_batch if r.peak_batch else '-':>5}"
        )
        print(row)


def run_full_benchmark(
    num_requests: int = 20,
    prompt_len: int = 64,
    output_len: int = 64,
    concurrency: int = 8,
    num_experts: int = 4,
    vocab_size: int = 32000,
):
    """Run all benchmarks and display results."""
    print("=" * 70)
    print("  vllm-i64 :: Engine Benchmark")
    print("=" * 70)

    results = []

    # Sync
    print("\n  [1/4] Sync engine...")
    r = bench_sync(num_requests, prompt_len, output_len, num_experts, vocab_size)
    results.append(r)
    print(f"        {r.throughput_tok_s:.0f} tok/s, TTFT {r.ttft_ms:.1f}ms")

    # Async with different concurrency
    for c in [1, concurrency]:
        label = f"async (c={c})"
        print(f"\n  [2/4] Async engine (concurrency={c})...")
        r = asyncio.run(bench_async(
            num_requests, prompt_len, output_len, c, num_experts, vocab_size
        ))
        results.append(r)
        print(f"        {r.throughput_tok_s:.0f} tok/s, peak batch {r.peak_batch}")

    # KV cache
    print(f"\n  [3/4] KV cache (lazy zeroing)...")
    kv = bench_kv_cache()
    print(f"        alloc+write {kv['alloc_write_ms']:.1f}ms, read {kv['read_ms']:.1f}ms, free {kv['free_ms']:.1f}ms")
    print(f"        dirty blocks after free: {kv['dirty_blocks']} (lazy — zeroed on next alloc)")

    # Routing (from existing bench)
    print(f"\n  [4/4] i64 routing vs float routing...")
    from benchmarks.bench_i64_routing import bench_i64_routing, bench_float_routing
    for n_tok in [1024, 4096]:
        r_i64 = bench_i64_routing(n_tok, num_experts)
        r_float = bench_float_routing(n_tok, num_experts)
        speedup = r_float['us_per_call'] / max(r_i64['us_per_call'], 0.01)
        print(f"        {n_tok} tokens: i64 {r_i64['us_per_call']}us vs float {r_float['us_per_call']}us ({speedup:.0f}x)")

    # Summary table
    print_results(results)
    print()
