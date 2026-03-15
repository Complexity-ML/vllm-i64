"""
KV Cache Correctness Test

Compares logits from:
  1. Full forward pass (no KV cache) — ground truth
  2. Prefill + decode with KV cache — should match

If logits diverge significantly, the KV cache is corrupting attention.

Usage:
    python test_kv_cache_correctness.py --model pacific-prime-python
"""

import argparse
import json
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from vllm_i64.core.loader import load_model_by_name
from vllm_i64.core.kv_cache import PagedKVCache


def load_model(model_name: str, dtype=torch.float32):
    """Load model from registry."""
    # Ensure relative paths resolve from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    old_cwd = os.getcwd()
    os.chdir(project_root)
    try:
        model = load_model_by_name(model_name, dtype=dtype, device="cpu")
    finally:
        os.chdir(old_cwd)
    model.eval()
    return model, model.config


def test_prefill_logits(model, config, token_ids, device="cpu"):
    """
    Test 1: Compare full prefill logits (no KV cache) vs prefill with KV cache.

    Both should produce identical logits for all positions.
    """
    print("\n=== Test 1: Prefill logits (no cache vs with cache) ===")

    n = len(token_ids)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.arange(n, dtype=torch.long, device=device)

    # Ground truth: no KV cache
    with torch.no_grad():
        logits_no_cache = model(
            token_ids=token_tensor,
            positions=positions,
            kv_cache=None,
            seq_ids=None,
            tokens_per_seq=None,
        )

    # With KV cache
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        block_size=16,
        num_blocks=64,
        max_seqs=4,
        dtype=torch.float32,
        device=device,
    )
    # Pre-allocate blocks for seq_id=0
    blocks_needed = (n + 15) // 16
    kv_cache.allocate_blocks(0, blocks_needed)

    with torch.no_grad():
        logits_with_cache = model(
            token_ids=token_tensor,
            positions=positions,
            kv_cache=kv_cache,
            seq_ids=[0],
            tokens_per_seq=[n],
        )

    # Compare
    diff = (logits_no_cache - logits_with_cache).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check argmax match (same top token at each position)
    top_no_cache = logits_no_cache.argmax(dim=-1)
    top_with_cache = logits_with_cache.argmax(dim=-1)
    argmax_match = (top_no_cache == top_with_cache).float().mean().item()

    print(f"  Max logit diff:    {max_diff:.6f}")
    print(f"  Mean logit diff:   {mean_diff:.6f}")
    print(f"  Argmax match rate: {argmax_match:.4f} ({int(argmax_match * n)}/{n})")

    ok = max_diff < 0.01 and argmax_match > 0.95
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok, kv_cache


def test_decode_logits(model, config, token_ids, kv_cache, device="cpu"):
    """
    Test 2: Compare decode logits.

    After prefilling N tokens, generate 1 more token.
    Compare full-sequence forward (N+1 tokens, no cache) vs
    decode step (1 token, position N, reading from KV cache).
    """
    print("\n=== Test 2: Decode logits (full forward vs cached decode) ===")

    n = len(token_ids)

    # Pick a plausible next token (argmax from prefill)
    with torch.no_grad():
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        positions = torch.arange(n, dtype=torch.long, device=device)
        prefill_logits = model(
            token_ids=token_tensor,
            positions=positions,
            kv_cache=None,
            seq_ids=None,
            tokens_per_seq=None,
        )
    next_token = prefill_logits[-1].argmax().item()
    print(f"  Next token (greedy): {next_token}")

    # Ground truth: full forward with N+1 tokens, no cache
    full_ids = token_ids + [next_token]
    full_tensor = torch.tensor(full_ids, dtype=torch.long, device=device)
    full_positions = torch.arange(n + 1, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_full = model(
            token_ids=full_tensor,
            positions=full_positions,
            kv_cache=None,
            seq_ids=None,
            tokens_per_seq=None,
        )
    # Last position logits from full forward
    logits_full_last = logits_full[-1]

    # Cached decode: 1 token at position N, using prefill KV cache
    # Ensure we have enough blocks
    blocks_needed = (n + 1 + 15) // 16
    current_blocks = len(kv_cache._seq_blocks.get(0, []))
    if blocks_needed > current_blocks:
        kv_cache.allocate_blocks(0, blocks_needed - current_blocks)

    decode_token = torch.tensor([next_token], dtype=torch.long, device=device)
    decode_pos = torch.tensor([n], dtype=torch.long, device=device)

    with torch.no_grad():
        logits_decode = model(
            token_ids=decode_token,
            positions=decode_pos,
            kv_cache=kv_cache,
            seq_ids=[0],
            tokens_per_seq=[1],
        )
    logits_decode_last = logits_decode[0]

    # Compare
    diff = (logits_full_last - logits_decode_last).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    top_full = logits_full_last.argmax().item()
    top_decode = logits_decode_last.argmax().item()
    argmax_match = top_full == top_decode

    print(f"  Max logit diff:    {max_diff:.6f}")
    print(f"  Mean logit diff:   {mean_diff:.6f}")
    print(f"  Top token (full):  {top_full}")
    print(f"  Top token (cache): {top_decode}")
    print(f"  Argmax match:      {'YES' if argmax_match else 'NO'}")

    ok = max_diff < 0.01 and argmax_match
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_multi_step_decode(model, config, token_ids, device="cpu", steps=5):
    """
    Test 3: Multi-step decode consistency.

    Generate `steps` tokens with KV cache, then do a full forward
    with all tokens (no cache) and compare final logits.
    """
    print(f"\n=== Test 3: Multi-step decode ({steps} steps) ===")

    n = len(token_ids)

    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        block_size=16,
        num_blocks=64,
        max_seqs=4,
        dtype=torch.float32,
        device=device,
    )
    blocks_needed = (n + steps + 15) // 16
    kv_cache.allocate_blocks(0, blocks_needed)

    # Prefill
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.arange(n, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            token_ids=token_tensor,
            positions=positions,
            kv_cache=kv_cache,
            seq_ids=[0],
            tokens_per_seq=[n],
        )

    generated = list(token_ids)

    # Decode steps
    for step in range(steps):
        next_token = logits[-1].argmax().item()
        generated.append(next_token)
        pos = n + step

        decode_token = torch.tensor([next_token], dtype=torch.long, device=device)
        decode_pos = torch.tensor([pos], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(
                token_ids=decode_token,
                positions=decode_pos,
                kv_cache=kv_cache,
                seq_ids=[0],
                tokens_per_seq=[1],
            )

    # Ground truth: full forward with all generated tokens, no cache
    full_tensor = torch.tensor(generated, dtype=torch.long, device=device)
    full_positions = torch.arange(len(generated), dtype=torch.long, device=device)

    with torch.no_grad():
        logits_full = model(
            token_ids=full_tensor,
            positions=full_positions,
            kv_cache=None,
            seq_ids=None,
            tokens_per_seq=None,
        )

    # Compare last position logits
    logits_cached_last = logits[0]
    logits_full_last = logits_full[-1]

    diff = (logits_full_last - logits_cached_last).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    top_full = logits_full_last.argmax().item()
    top_decode = logits_cached_last.argmax().item()
    argmax_match = top_full == top_decode

    print(f"  Generated {steps} tokens: {generated[n:]}")
    print(f"  Max logit diff:    {max_diff:.6f}")
    print(f"  Mean logit diff:   {mean_diff:.6f}")
    print(f"  Top token (full):  {top_full}")
    print(f"  Top token (cache): {top_decode}")
    print(f"  Argmax match:      {'YES' if argmax_match else 'NO'}")

    ok = max_diff < 0.05 and argmax_match
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="KV Cache Correctness Test")
    parser.add_argument("--model", default="pacific-prime-python")
    parser.add_argument("--prompt", default="def fibonacci(n):\n    ")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading model: {args.model} (dtype={args.dtype})")
    model, config = load_model(args.model, dtype=dtype)
    print(f"  Layers: {config.num_hidden_layers}, KV heads: {config.num_key_value_heads}, head_dim: {config.head_dim}")

    # Tokenize prompt
    from vllm_i64.core.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(args.model)
    if tokenizer is None:
        print("ERROR: No tokenizer found")
        sys.exit(1)

    token_ids = tokenizer.encode(args.prompt)
    print(f"  Prompt: {repr(args.prompt)}")
    print(f"  Token IDs ({len(token_ids)}): {token_ids}")

    results = {}

    # Test 1: Prefill
    ok1, kv_cache = test_prefill_logits(model, config, token_ids)
    results["prefill"] = "PASS" if ok1 else "FAIL"

    # Test 2: Single decode step
    ok2 = test_decode_logits(model, config, token_ids, kv_cache)
    results["decode_single"] = "PASS" if ok2 else "FAIL"

    # Test 3: Multi-step decode
    ok3 = test_multi_step_decode(model, config, token_ids, steps=args.steps)
    results["decode_multi"] = "PASS" if ok3 else "FAIL"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = all(v == "PASS" for v in results.values())
    for test, result in results.items():
        print(f"  {test:20s}: {result}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

    if not all_pass:
        print("\n  The KV cache is producing different logits than a full forward pass.")
        print("  This confirms an inference regression in the paged KV cache.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
