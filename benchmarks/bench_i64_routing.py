"""
vllm-i64 :: Routing Benchmark

Compare i64 routing vs float-based MoE routing.

i64 routing: token_id & mask → 1 integer op per token
Float MoE:   hidden @ W_gate → softmax → topk → ~3 matmuls + softmax per token

INL - 2025
"""

import torch
import time
import numpy as np
from vllm_i64.kernels.i64_ops import i64_route_tokens, i64_full_pipeline


def bench_i64_routing(num_tokens: int, num_experts: int, n_iters: int = 1000):
    """Benchmark pure i64 routing."""
    token_ids = torch.randint(0, 100_000, (num_tokens,), dtype=torch.int64)

    # Warmup
    for _ in range(10):
        i64_route_tokens(token_ids, num_experts)

    start = time.perf_counter()
    for _ in range(n_iters):
        expert_ids = i64_route_tokens(token_ids, num_experts)
    elapsed = time.perf_counter() - start

    us_per_call = elapsed / n_iters * 1e6
    ns_per_token = elapsed / n_iters / num_tokens * 1e9

    return {
        "method": "i64_bit_mask",
        "num_tokens": num_tokens,
        "us_per_call": round(us_per_call, 2),
        "ns_per_token": round(ns_per_token, 2),
    }


def bench_float_routing(num_tokens: int, num_experts: int, hidden_dim: int = 768, n_iters: int = 1000):
    """Benchmark traditional float-based MoE routing (for comparison)."""
    hidden = torch.randn(num_tokens, hidden_dim)
    gate_weight = torch.randn(hidden_dim, num_experts)

    # Warmup
    for _ in range(10):
        logits = hidden @ gate_weight
        probs = torch.softmax(logits, dim=-1)
        expert_ids = probs.argmax(dim=-1)

    start = time.perf_counter()
    for _ in range(n_iters):
        logits = hidden @ gate_weight
        probs = torch.softmax(logits, dim=-1)
        expert_ids = probs.argmax(dim=-1)
    elapsed = time.perf_counter() - start

    us_per_call = elapsed / n_iters * 1e6
    ns_per_token = elapsed / n_iters / num_tokens * 1e9

    return {
        "method": "float_softmax_gate",
        "num_tokens": num_tokens,
        "us_per_call": round(us_per_call, 2),
        "ns_per_token": round(ns_per_token, 2),
    }


def bench_full_pipeline(num_tokens: int, hidden_dim: int = 768, num_experts: int = 4, n_iters: int = 100):
    """Benchmark full i64 pipeline: route → scatter → expert → gather."""
    expert_inter = hidden_dim // num_experts

    token_ids = torch.randint(0, 100_000, (num_tokens,), dtype=torch.int64)
    hidden = torch.randn(num_tokens, hidden_dim)
    gate_up_w = torch.randn(num_experts, hidden_dim, 2 * expert_inter) * 0.02
    down_w = torch.randn(num_experts, expert_inter, hidden_dim) * 0.02

    # Warmup
    for _ in range(5):
        i64_full_pipeline(token_ids, hidden, gate_up_w, down_w, num_experts)

    start = time.perf_counter()
    for _ in range(n_iters):
        output = i64_full_pipeline(token_ids, hidden, gate_up_w, down_w, num_experts)
    elapsed = time.perf_counter() - start

    ms_per_call = elapsed / n_iters * 1e3
    tokens_per_sec = num_tokens * n_iters / elapsed

    return {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
        "ms_per_call": round(ms_per_call, 2),
        "tokens_per_sec": int(tokens_per_sec),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-i64 :: Routing Benchmark")
    print("=" * 60)

    # --- Routing comparison ---
    print("\n--- i64 routing vs float routing ---")
    print(f"{'Method':<25} {'Tokens':>8} {'us/call':>10} {'ns/tok':>10}")
    print("-" * 55)

    for n_tok in [256, 1024, 4096, 16384]:
        r_i64 = bench_i64_routing(n_tok, num_experts=4)
        r_float = bench_float_routing(n_tok, num_experts=4)

        print(f"{r_i64['method']:<25} {n_tok:>8} {r_i64['us_per_call']:>10} {r_i64['ns_per_token']:>10}")
        print(f"{r_float['method']:<25} {n_tok:>8} {r_float['us_per_call']:>10} {r_float['ns_per_token']:>10}")
        speedup = r_float['us_per_call'] / max(r_i64['us_per_call'], 0.01)
        print(f"{'  → speedup':<25} {'':>8} {f'{speedup:.0f}x':>10}")
        print()

    # --- Full pipeline ---
    print("\n--- Full i64 pipeline (route → scatter → expert → gather) ---")
    print(f"{'Tokens':>8} {'Hidden':>8} {'Experts':>8} {'ms/call':>10} {'tok/s':>12}")
    print("-" * 50)

    for n_tok in [256, 1024, 4096]:
        for n_exp in [4, 8]:
            r = bench_full_pipeline(n_tok, hidden_dim=768, num_experts=n_exp)
            print(f"{r['num_tokens']:>8} {r['hidden_dim']:>8} {r['num_experts']:>8} "
                  f"{r['ms_per_call']:>10} {r['tokens_per_sec']:>12,}")

    print("\nDone.")
