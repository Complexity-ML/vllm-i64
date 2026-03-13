"""
Ablation Eval — run inference on each of the 4 ablation runs sequentially.

Usage:
    python eval_ablations.py
    python eval_ablations.py --port 8001 --dtype float32
    python eval_ablations.py --prompts prompts.txt

Each model is started, queried with every prompt, then shut down before
the next model loads.  Results are written to ablation_results.json.

INL - 2025
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ---------------------------------------------------------------------------
# Models to evaluate (in order)
# ---------------------------------------------------------------------------
ABLATION_MODELS = [
    "run2-full",
    "pacific-tiny-chat",
]

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ablation-150m")

# Pretrain completion prompts (FineWeb-Edu style — text continuation, not Q&A)
DEFAULT_PROMPTS = [
    # --- factual continuation ---
    "The Eiffel Tower was built in",
    "In mathematics, a prime number is defined as",
    "The transformer architecture, introduced in the paper 'Attention is All You Need',",
    # --- educational text continuation ---
    "Machine learning is a subfield of artificial intelligence that focuses on",
    "The water cycle, also known as the hydrological cycle, describes how",
    # --- code completion ---
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1:\n",
    # --- science ---
    "Photosynthesis is the process by which plants convert sunlight into",
    # --- longer reasoning seed ---
    "The main difference between supervised and unsupervised learning is that",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    prompt: str
    response: str
    latency_ms: float
    tokens_generated: int


@dataclass
class ModelResult:
    model: str
    results: List[PromptResult] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(model: str, port: int, dtype: str, no_cuda_graphs: bool) -> subprocess.Popen:
    # pacific-tiny-chat lives at repo root, ablation runs in CHECKPOINT_DIR
    candidate = os.path.abspath(os.path.join(CHECKPOINT_DIR, model, "final"))
    if os.path.isdir(candidate):
        ckpt = candidate
    else:
        ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model))
    cmd = [
        sys.executable, "-m", "vllm_i64.cli",
        "serve", model,
        "--checkpoint", ckpt,
        "--port", str(port),
        "--dtype", dtype,
        "--quantization", "none",
    ]
    if no_cuda_graphs:
        cmd.append("--no-cuda-graphs")

    print(f"  Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


def wait_for_health(port: int, timeout: int = 120) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_completion(port: int, prompt: str, max_tokens: int, temperature: float) -> tuple[str, int]:
    """Returns (response_text, tokens_generated)."""
    url = f"http://localhost:{port}/v1/completions"
    payload = json.dumps({
        "model": "ablation",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "min_p": 0.05,           # match website: cuts improbable tokens
        "typical_p": 0.92,       # match website: entropy-based filter — kills drift
        "repetition_penalty": 1.4,
        "min_tokens": 8,         # match website: no premature EOS
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())

    text = data["choices"][0]["text"]
    tokens = data["usage"]["completion_tokens"]
    return text, tokens


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def eval_model(
    model: str,
    prompts: List[str],
    port: int,
    max_tokens: int,
    temperature: float,
    dtype: str,
    no_cuda_graphs: bool,
) -> ModelResult:
    result = ModelResult(model=model)

    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    proc = start_server(model, port, dtype, no_cuda_graphs)

    try:
        print("  Waiting for server to be healthy...", end="", flush=True)
        if not wait_for_health(port):
            print(" TIMEOUT")
            result.errors = len(prompts)
            return result
        print(" OK")

        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] {prompt[:70]}...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                text, tokens = run_completion(port, prompt, max_tokens, temperature)
                latency_ms = (time.perf_counter() - t0) * 1000
                result.results.append(PromptResult(
                    prompt=prompt,
                    response=text.strip(),
                    latency_ms=round(latency_ms, 1),
                    tokens_generated=tokens,
                ))
                print(f" {latency_ms:.0f}ms, {tokens}tok")
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                print(f" ERROR: {e}")
                result.results.append(PromptResult(
                    prompt=prompt,
                    response=f"ERROR: {e}",
                    latency_ms=round(latency_ms, 1),
                    tokens_generated=0,
                ))
                result.errors += 1

    finally:
        print(f"  Stopping server...", end="", flush=True)
        stop_server(proc)
        print(" done")

    ok_results = [r for r in result.results if not r.response.startswith("ERROR")]
    if ok_results:
        result.avg_latency_ms = round(sum(r.latency_ms for r in ok_results) / len(ok_results), 1)
        result.total_tokens = sum(r.tokens_generated for r in ok_results)

    return result


def print_summary(all_results: List[ModelResult], prompts: List[str]):
    print(f"\n{'='*70}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")

    # Per-model stats
    print(f"\n{'Model':<20} {'Prompts OK':<12} {'Avg Latency':>12} {'Total Tokens':>14}")
    print("-" * 60)
    for r in all_results:
        ok = len(r.results) - r.errors
        print(f"  {r.model:<18} {ok}/{len(r.results):<10} {r.avg_latency_ms:>10.0f}ms {r.total_tokens:>13}")

    # Per-prompt comparison
    print(f"\n{'─'*70}")
    print("  RESPONSES PER PROMPT")
    print(f"{'─'*70}")
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt[:80]}")
        for r in all_results:
            if i < len(r.results):
                resp = r.results[i].response[:120].replace("\n", " ")
                print(f"  [{r.model:<15}] {resp}")


def main():
    parser = argparse.ArgumentParser(description="Sequential ablation inference eval")
    parser.add_argument("--port", type=int, default=8099, help="Server port (default: 8099)")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "bfloat16", "float32"],
                        help="Weight dtype (default: float32)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompts", default=None, help="Path to prompts file (one per line)")
    parser.add_argument("--output", default="ablation_results.json")
    parser.add_argument("--no-cuda-graphs", action="store_true",
                        help="Disable CUDA graphs (use on shared GPU or CPU)")
    parser.add_argument("--models", nargs="+", default=ABLATION_MODELS,
                        help="Models to evaluate (default: all 4 ablations)")
    args = parser.parse_args()

    # Load prompts
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    print(f"Ablation eval: {len(args.models)} models × {len(prompts)} prompts")
    print(f"  dtype={args.dtype}  max_tokens={args.max_tokens}  port={args.port}")
    print(f"  output={args.output}")

    all_results: List[ModelResult] = []

    for model in args.models:
        result = eval_model(
            model=model,
            prompts=prompts,
            port=args.port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            dtype=args.dtype,
            no_cuda_graphs=args.no_cuda_graphs,
        )
        all_results.append(result)

        # Small gap between models so port is fully released
        time.sleep(2)

    print_summary(all_results, prompts)

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
