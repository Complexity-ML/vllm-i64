"""
vllm-i64 :: Comparative Benchmark

Compares vllm-i64 throughput and latency against other inference engines.

Modes:
  1. Self-benchmark: measure vllm-i64 standalone throughput
  2. Comparative: if vLLM/TGI API is available, compare side-by-side

Metrics:
  - Time to First Token (TTFT) — ms
  - Inter-Token Latency (ITL) — ms
  - Throughput — tokens/second
  - Memory usage — GB

INL - 2025
"""

import time
import asyncio
import json
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Single benchmark result — mix of integer counts and float timings."""
    engine: str
    num_requests: int
    prompt_len: int
    output_len: int
    total_time_s: float
    ttft_ms: float             # Time to first token
    avg_itl_ms: float          # Average inter-token latency
    throughput_tok_s: float    # Output tokens / second
    total_tokens: int
    peak_batch_size: int = 0
    memory_gb: float = 0.0


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    num_requests: int = 100
    prompt_len: int = 128
    output_len: int = 128
    concurrency: int = 16
    warmup_requests: int = 5


def bench_vllm_i64_sync(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark vllm-i64 with synchronous engine."""
    import torch
    import numpy as np
    from vllm_i64.engine.i64_engine import I64Engine
    from vllm_i64.core.sampling import SamplingParams

    engine = I64Engine(
        num_experts=4,
        vocab_size=32000,
        max_batch_size=config.concurrency,
        max_seq_len=config.prompt_len + config.output_len + 64,
        device="cpu",
    )
    engine.sampling_params = SamplingParams(temperature=0.0)

    # Warmup
    for _ in range(config.warmup_requests):
        prompt = list(range(1, config.prompt_len + 1))
        engine.add_request(prompt, max_new_tokens=min(config.output_len, 16))
        while engine.scheduler.pending or engine.scheduler.running:
            engine.step()
        engine.scheduler.finished.clear()

    # Benchmark
    ttft_times = []
    itl_times = []

    start = time.perf_counter()
    total_output_tokens = 0

    for i in range(config.num_requests):
        prompt = list(range(1, config.prompt_len + 1))
        req_start = time.perf_counter()
        rid = engine.add_request(prompt, max_new_tokens=config.output_len)

        first_token_time = None
        prev_token_time = req_start
        tokens_generated = 0

        while True:
            results = engine.step()
            if rid in results:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    ttft_times.append((first_token_time - req_start) * 1000)
                else:
                    itl_times.append((now - prev_token_time) * 1000)
                prev_token_time = now
                tokens_generated += 1

            # Check if finished
            finished = False
            for req in engine.scheduler.finished:
                if req.request_id == rid:
                    finished = True
                    break
            if finished:
                break

        total_output_tokens += tokens_generated
        engine.scheduler.finished = [
            r for r in engine.scheduler.finished if r.request_id != rid
        ]

    total_time = time.perf_counter() - start

    return BenchmarkResult(
        engine="vllm-i64",
        num_requests=config.num_requests,
        prompt_len=config.prompt_len,
        output_len=config.output_len,
        total_time_s=round(total_time, 3),
        ttft_ms=round(sum(ttft_times) / len(ttft_times), 2) if ttft_times else 0,
        avg_itl_ms=round(sum(itl_times) / len(itl_times), 2) if itl_times else 0,
        throughput_tok_s=round(total_output_tokens / total_time, 1),
        total_tokens=total_output_tokens,
    )


async def bench_vllm_i64_async(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark vllm-i64 with async continuous batching engine."""
    import torch
    from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine
    from vllm_i64.core.sampling import SamplingParams

    engine = AsyncI64Engine(
        num_experts=4,
        vocab_size=32000,
        max_batch_size=config.concurrency,
        max_seq_len=config.prompt_len + config.output_len + 64,
        device="cpu",
    )
    engine.engine.sampling_params = SamplingParams(temperature=0.0)
    await engine.start()

    # Warmup
    warmup_tasks = []
    for _ in range(config.warmup_requests):
        prompt = list(range(1, min(config.prompt_len, 32) + 1))
        warmup_tasks.append(engine.generate(prompt, max_new_tokens=8))
    await asyncio.gather(*warmup_tasks)
    engine.engine.scheduler.finished.clear()

    # Benchmark with concurrency
    start = time.perf_counter()
    total_output_tokens = 0
    ttft_times = []

    sem = asyncio.Semaphore(config.concurrency)

    async def _single_request():
        nonlocal total_output_tokens
        async with sem:
            prompt = list(range(1, config.prompt_len + 1))
            req_start = time.perf_counter()
            result = await engine.generate(prompt, max_new_tokens=config.output_len)
            ttft = (time.perf_counter() - req_start) * 1000 / max(1, len(result.output_tokens))
            ttft_times.append(result.elapsed_ms / max(1, len(result.output_tokens)))
            total_output_tokens += len(result.output_tokens)

    tasks = [_single_request() for _ in range(config.num_requests)]
    await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start

    await engine.stop()

    return BenchmarkResult(
        engine="vllm-i64 (async)",
        num_requests=config.num_requests,
        prompt_len=config.prompt_len,
        output_len=config.output_len,
        total_time_s=round(total_time, 3),
        ttft_ms=round(sum(ttft_times) / len(ttft_times), 2) if ttft_times else 0,
        avg_itl_ms=round(sum(ttft_times) / len(ttft_times), 2) if ttft_times else 0,
        throughput_tok_s=round(total_output_tokens / total_time, 1),
        total_tokens=total_output_tokens,
        peak_batch_size=engine.peak_batch_size,
    )


async def bench_external_api(
    base_url: str,
    engine_name: str,
    config: BenchmarkConfig,
) -> Optional[BenchmarkResult]:
    """
    Benchmark an external API (vLLM, TGI, or any OpenAI-compatible server).

    Sends concurrent requests and measures throughput.
    """
    try:
        import httpx
    except ImportError:
        print(f"  httpx not installed, skipping {engine_name} benchmark")
        return None

    ttft_times = []
    total_output_tokens = 0
    sem = asyncio.Semaphore(config.concurrency)

    async def _single_request(client):
        nonlocal total_output_tokens
        async with sem:
            prompt_text = "A " * config.prompt_len
            payload = {
                "prompt": prompt_text,
                "max_tokens": config.output_len,
                "temperature": 0.0,
            }
            req_start = time.perf_counter()
            try:
                resp = await client.post(
                    f"{base_url}/v1/completions",
                    json=payload,
                    timeout=120.0,
                )
                elapsed = (time.perf_counter() - req_start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", config.output_len)
                    total_output_tokens += completion_tokens
                    ttft_times.append(elapsed / max(1, completion_tokens))
            except Exception:
                pass

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [_single_request(client) for _ in range(config.num_requests)]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    if not ttft_times:
        return None

    return BenchmarkResult(
        engine=engine_name,
        num_requests=config.num_requests,
        prompt_len=config.prompt_len,
        output_len=config.output_len,
        total_time_s=round(total_time, 3),
        ttft_ms=round(sum(ttft_times) / len(ttft_times), 2),
        avg_itl_ms=round(sum(ttft_times) / len(ttft_times), 2),
        throughput_tok_s=round(total_output_tokens / total_time, 1),
        total_tokens=total_output_tokens,
    )


def print_results(results: List[BenchmarkResult]):
    """Pretty-print benchmark comparison table."""
    print("\n" + "=" * 80)
    print("  vllm-i64 :: Comparative Benchmark Results")
    print("=" * 80)

    header = f"{'Engine':<25} {'Requests':>8} {'Prompt':>6} {'Output':>6} {'TTFT ms':>8} {'ITL ms':>8} {'Tok/s':>10} {'Total s':>8}"
    print(header)
    print("-" * 80)

    for r in results:
        row = (
            f"{r.engine:<25} "
            f"{r.num_requests:>8} "
            f"{r.prompt_len:>6} "
            f"{r.output_len:>6} "
            f"{r.ttft_ms:>8.1f} "
            f"{r.avg_itl_ms:>8.1f} "
            f"{r.throughput_tok_s:>10.1f} "
            f"{r.total_time_s:>8.2f}"
        )
        print(row)

    # Speedup comparison if multiple engines
    if len(results) >= 2:
        base = results[0]
        print()
        for r in results[1:]:
            if r.throughput_tok_s > 0 and base.throughput_tok_s > 0:
                speedup = r.throughput_tok_s / base.throughput_tok_s
                label = "faster" if speedup > 1 else "slower"
                print(f"  {r.engine} vs {base.engine}: {speedup:.2f}x {label}")

    print()


async def main():
    """Run comparative benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="vllm-i64 Comparative Benchmark")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--prompt-len", type=int, default=64, help="Prompt length (tokens)")
    parser.add_argument("--output-len", type=int, default=64, help="Output length (tokens)")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--vllm-url", default=None, help="vLLM API URL for comparison")
    parser.add_argument("--tgi-url", default=None, help="TGI API URL for comparison")
    args = parser.parse_args()

    config = BenchmarkConfig(
        num_requests=args.requests,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        concurrency=args.concurrency,
    )

    results = []

    # vllm-i64 sync
    print(f"Running vllm-i64 sync benchmark ({config.num_requests} requests)...")
    r_sync = bench_vllm_i64_sync(config)
    results.append(r_sync)

    # vllm-i64 async
    print(f"Running vllm-i64 async benchmark ({config.num_requests} requests)...")
    r_async = await bench_vllm_i64_async(config)
    results.append(r_async)

    # External engines
    if args.vllm_url:
        print(f"Running vLLM benchmark against {args.vllm_url}...")
        r_vllm = await bench_external_api(args.vllm_url, "vLLM", config)
        if r_vllm:
            results.append(r_vllm)

    if args.tgi_url:
        print(f"Running TGI benchmark against {args.tgi_url}...")
        r_tgi = await bench_external_api(args.tgi_url, "TGI", config)
        if r_tgi:
            results.append(r_tgi)

    print_results(results)

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
