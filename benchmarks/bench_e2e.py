"""
vllm-i64 :: End-to-End Benchmark

Measures real inference throughput:
  - Prompt processing (prefill) tok/s
  - Token generation (decode) tok/s
  - Total latency per request
  - Batch throughput

INL - 2025
"""

import torch
import time
import numpy as np
from typing import List, Optional


def bench_prefill(
    model,
    prompt_lengths: List[int] = [128, 256, 512, 1024],
    num_experts: int = 4,
    vocab_size: int = 32000,
    n_iters: int = 10,
    device: str = "cpu",
) -> List[dict]:
    """Benchmark prefill (prompt processing) throughput."""
    results = []

    for seq_len in prompt_lengths:
        token_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int64, device=device)
        positions = torch.arange(seq_len, dtype=torch.int32, device=device)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                model(token_ids, positions)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                model(token_ids, positions)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        ms_per_call = elapsed / n_iters * 1000
        tok_per_sec = seq_len * n_iters / elapsed

        results.append({
            "phase": "prefill",
            "seq_len": seq_len,
            "ms_per_call": round(ms_per_call, 2),
            "tok_per_sec": int(tok_per_sec),
        })

    return results


def bench_decode(
    model,
    batch_sizes: List[int] = [1, 4, 8, 16],
    num_experts: int = 4,
    vocab_size: int = 32000,
    num_steps: int = 50,
    device: str = "cpu",
) -> List[dict]:
    """Benchmark decode (single token generation) throughput."""
    results = []

    for bsz in batch_sizes:
        token_ids = torch.randint(0, vocab_size, (bsz,), dtype=torch.int64, device=device)
        positions = torch.randint(0, 1024, (bsz,), dtype=torch.int32, device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(token_ids, positions)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_steps):
            with torch.no_grad():
                logits = model(token_ids, positions)
            token_ids = logits.argmax(dim=-1)
            positions = positions + 1

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        ms_per_step = elapsed / num_steps * 1000
        tok_per_sec = bsz * num_steps / elapsed

        results.append({
            "phase": "decode",
            "batch_size": bsz,
            "ms_per_step": round(ms_per_step, 2),
            "tok_per_sec": int(tok_per_sec),
        })

    return results


def bench_generation(
    engine,
    prompts: List[List[int]],
    max_new_tokens: int = 128,
) -> dict:
    """Benchmark full generation through the engine."""
    start = time.perf_counter()

    total_output = 0
    total_prompt = 0

    for prompt_ids in prompts:
        result = engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
        )
        total_output += len(result.output_tokens)
        total_prompt += len(prompt_ids)

    elapsed = time.perf_counter() - start

    return {
        "num_requests": len(prompts),
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "elapsed_s": round(elapsed, 3),
        "prompt_tok_per_sec": int(total_prompt / elapsed),
        "output_tok_per_sec": int(total_output / elapsed),
        "total_tok_per_sec": int((total_prompt + total_output) / elapsed),
    }


if __name__ == "__main__":
    from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
    from vllm_i64.models.complexity_deep.model import ComplexityDeepModel
    from vllm_i64.engine.i64_engine import I64Engine

    print("=" * 60)
    print("vllm-i64 :: End-to-End Benchmark")
    print("=" * 60)

    # Small model for CPU benchmarking
    config = ComplexityDeepConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1408,
        num_experts=4,
    )
    model = ComplexityDeepModel(config)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")

    # Prefill
    print("\n--- Prefill ---")
    print(f"{'SeqLen':>8} {'ms/call':>10} {'tok/s':>12}")
    print("-" * 35)
    for r in bench_prefill(model, device=device):
        print(f"{r['seq_len']:>8} {r['ms_per_call']:>10} {r['tok_per_sec']:>12,}")

    # Decode
    print("\n--- Decode ---")
    print(f"{'Batch':>8} {'ms/step':>10} {'tok/s':>12}")
    print("-" * 35)
    for r in bench_decode(model, device=device):
        print(f"{r['batch_size']:>8} {r['ms_per_step']:>10} {r['tok_per_sec']:>12,}")

    # Engine generation
    print("\n--- Engine Generation ---")
    engine = I64Engine(model=None, num_experts=4, vocab_size=config.vocab_size)
    prompts = [
        list(range(10, 10 + length))
        for length in [32, 64, 128, 256]
    ]
    r = bench_generation(engine, prompts, max_new_tokens=64)
    print(f"  Requests:    {r['num_requests']}")
    print(f"  Prompt tok:  {r['total_prompt_tokens']}")
    print(f"  Output tok:  {r['total_output_tokens']}")
    print(f"  Elapsed:     {r['elapsed_s']}s")
    print(f"  Output tok/s: {r['output_tok_per_sec']:,}")

    print("\nDone.")
