"""
vllm-i64 :: CLI

Usage:
    vllm-i64 serve <model> [--port 8000] [--host 0.0.0.0] [--dtype float16] [--tp 1]
    vllm-i64 bench [--mode all|routing|engine] [--requests 20] [--concurrency 8]
    vllm-i64 list
    vllm-i64 check <model>

Multi-GPU:
    vllm-i64 serve pacific-prime-chat --tp 4
    → launches 4 workers via torchrun, continuous batching on each

INL - 2025
"""

import argparse
import logging
import sys

_logger = logging.getLogger("vllm_i64.cli")


def cmd_serve(args):
    """Start inference server (single-GPU or multi-GPU via torchrun)."""

    # If TP > 1 or PP > 1, launch via torchrun
    if args.tp > 1 or args.pp > 1:
        # Disaggregated prefill/decode: split GPUs into prefill and decode groups
        if getattr(args, 'disaggregated', False) and args.tp >= 2:
            from vllm_i64.parallel.disaggregated import launch_disaggregated

            forward_args = ["serve", args.model]
            forward_args += ["--host", args.host]
            forward_args += ["--port", str(args.port)]
            forward_args += ["--dtype", args.dtype]
            forward_args += ["--disaggregated"]
            if args.checkpoint:
                forward_args += ["--checkpoint", args.checkpoint]
            if args.chat_template:
                forward_args += ["--chat-template", args.chat_template]
            if args.quantization and args.quantization != "none":
                forward_args += ["--quantization", args.quantization]

            _logger.info("Disaggregated mode: GPU 0 = prefill, GPU 1 = decode")
            rc = launch_disaggregated(tp_size=args.tp, args=forward_args)
            sys.exit(rc)

        from vllm_i64.parallel.launcher import launch_distributed

        # Forward all args to the worker
        forward_args = ["serve", args.model]
        forward_args += ["--host", args.host]
        forward_args += ["--port", str(args.port)]
        forward_args += ["--dtype", args.dtype]
        if args.checkpoint:
            forward_args += ["--checkpoint", args.checkpoint]
        if args.chat_template:
            forward_args += ["--chat-template", args.chat_template]
        if args.quantization and args.quantization != "none":
            forward_args += ["--quantization", args.quantization]

        rc = launch_distributed(tp_size=args.tp, pp_size=args.pp, args=forward_args)
        sys.exit(rc)

    # Single-GPU: run directly
    import torch
    from vllm_i64.core.logging import setup_logging
    setup_logging(level="INFO")
    from vllm_i64.core.loader import load_model_by_name
    from vllm_i64.engine.i64_engine import I64Engine
    from vllm_i64.cpu.engine import CPUEngine
    from vllm_i64.api.server import I64Server
    from vllm_i64.core.tokenizer import load_tokenizer
    from vllm_i64.core.registry import get_model_entry

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CPU doesn't support FP16/BF16 natively on most hardware — override to float32
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        _logger.info("CPU detected — overriding %s to float32", dtype)
        dtype = torch.float32

    # INT8 on CPU is slower than float32 — quantize/dequant overhead per layer kills throughput
    if device == "cpu" and args.quantization == "int8":
        _logger.info("CPU detected — disabling INT8 quantization (float32 is faster)")
        args.quantization = "none"

    entry = get_model_entry(args.model)
    _logger.info("vllm-i64 :: serving %s", args.model)
    _logger.info("host=%s port=%d dtype=%s device=%s", args.host, args.port, dtype, device)

    # Load model
    model = load_model_by_name(
        args.model, dtype=dtype, device=device,
        checkpoint_override=args.checkpoint,
        quantization=args.quantization,
    )
    model.eval()

    # torch.compile for kernel fusion (CPU + GPU)
    if getattr(args, 'compile', False):
        compile_mode = getattr(args, 'compile_mode', 'reduce-overhead')
        # CPU backend doesn't support reduce-overhead (CUDA graphs)
        if device == "cpu" and compile_mode == "reduce-overhead":
            compile_mode = "default"
        _logger.info("torch.compile: mode=%s", compile_mode)
        model = torch.compile(model, mode=compile_mode)

    # Load tokenizer (from checkpoint dir if overridden, search up 3 parent dirs)
    tokenizer = None
    if args.checkpoint:
        import os
        search_dir = args.checkpoint
        for _ in range(4):  # checkpoint, parent, grandparent, great-grandparent
            tok_path = os.path.join(search_dir, "tokenizer.json")
            if os.path.exists(tok_path):
                from vllm_i64.core.tokenizer import I64Tokenizer
                tokenizer = I64Tokenizer(tok_path)
                _logger.info("tokenizer: %s", tok_path)
                break
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent
    if tokenizer is None:
        tokenizer = load_tokenizer(args.model)

    # Load chat template (explicit path or auto-detect from checkpoint dir)
    chat_template = None
    if args.chat_template:
        with open(args.chat_template) as f:
            chat_template = f.read()
    else:
        import os
        ckpt_path = args.checkpoint or entry.checkpoint
        if ckpt_path:
            search_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else ckpt_path
            for _ in range(4):  # checkpoint, parent, grandparent, great-grandparent
                for name in ("chat_template.jinja", "chat_template.j2"):
                    tmpl_path = os.path.join(search_dir, name)
                    if os.path.exists(tmpl_path):
                        with open(tmpl_path) as f:
                            chat_template = f.read()
                        _logger.info("chat_template: %s", tmpl_path)
                        break
                if chat_template:
                    break
                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    break
                search_dir = parent

    # Create engine + server (async continuous batching)
    # CPU uses the dedicated CPUEngine (no CUDA graphs, thread-executor step)
    # GPU uses the full I64Engine (CUDA graphs, Triton, FP8, etc.)
    common_kwargs = dict(
        model=model,
        num_experts=getattr(model.config, 'num_experts', 1),
        vocab_size=model.config.vocab_size,
        enable_prefix_caching=getattr(args, 'enable_prefix_caching', False),
        kv_cache_dtype=getattr(args, 'kv_cache_dtype', None),
        max_kv_blocks=getattr(args, 'max_kv_blocks', 0),
    )
    if device == "cpu":
        engine = CPUEngine(**common_kwargs)
    else:
        engine = I64Engine(**common_kwargs, device=device)

    # Enable swap-to-CPU for KV cache overflow
    if getattr(args, 'enable_swap', False) and engine.kv_cache is not None:
        engine.kv_cache.enable_swap()
        _logger.info("swap-to-cpu: enabled")

    # Speculative decoding (optional)
    if getattr(args, 'speculative_model', None):
        draft_model = load_model_by_name(args.speculative_model, dtype=dtype, device=device)
        draft_model.eval()
        num_spec = getattr(args, 'num_speculative_tokens', 5)
        engine.enable_speculative(draft_model, num_spec)
        _logger.info("speculative: %s (K=%d)", args.speculative_model, num_spec)

    # CUDA graph warmup (captures decode graphs for common batch sizes)
    if not getattr(args, 'no_cuda_graphs', False):
        engine.warmup_and_capture_graphs()
    server = I64Server(
        engine=engine,
        tokenizer=tokenizer,
        chat_template=chat_template,
        model_name=args.model,
        host=args.host,
        port=args.port,
        api_key=getattr(args, 'api_key', None),
        rate_limit=getattr(args, 'rate_limit', 0),
        max_pending=getattr(args, 'max_pending', 0),
        rag_index_path=getattr(args, 'rag_index', None),
    )
    server.run()


def cmd_list(args):
    """List registered models."""
    from vllm_i64.core.registry import list_models

    models = list_models()
    if not models:
        print("No models registered.")
        return

    print(f"{'Name':<25} {'Params':>10} {'Description'}")
    print("-" * 60)
    for m in models:
        print(f"{m['name']:<25} {m['parameters']:>10} {m['description']}")


def cmd_check(args):
    """Check model: load config, verify checkpoint exists."""
    from vllm_i64.core.registry import get_model_entry
    import os

    entry = get_model_entry(args.model)
    print(f"Model:       {entry.name}")
    print(f"Class:       {entry.model_class}")
    print(f"Parameters:  {entry.parameters}")
    print(f"Config:      {entry.config_path}")
    print(f"Checkpoint:  {entry.checkpoint}")

    if entry.config_path and os.path.exists(entry.config_path):
        print(f"  config.json  OK")
    else:
        print(f"  config.json  MISSING")

    if entry.checkpoint and os.path.exists(entry.checkpoint):
        size_mb = os.path.getsize(entry.checkpoint) / 1e6 if os.path.isfile(entry.checkpoint) else 0
        print(f"  checkpoint   OK ({size_mb:.0f} MB)" if size_mb else "  checkpoint   OK (directory)")
    else:
        print(f"  checkpoint   MISSING")


def cmd_bench(args):
    """Run benchmarks."""
    import os
    # benchmarks/ lives at project root, not inside vllm_i64 package
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.mode == "routing":
        from benchmarks.bench_i64_routing import (
            bench_i64_routing,
            bench_float_routing,
            bench_full_pipeline,
        )

        print("=" * 60)
        print("vllm-i64 :: Routing Benchmark")
        print("=" * 60)

        print(f"\n--- i64 routing vs float routing ---")
        print(f"{'Method':<25} {'Tokens':>8} {'us/call':>10} {'ns/tok':>10}")
        print("-" * 55)

        for n_tok in [256, 1024, 4096]:
            r_i64 = bench_i64_routing(n_tok, num_experts=args.num_experts)
            r_float = bench_float_routing(n_tok, num_experts=args.num_experts)

            print(f"{r_i64['method']:<25} {n_tok:>8} {r_i64['us_per_call']:>10} {r_i64['ns_per_token']:>10}")
            print(f"{r_float['method']:<25} {n_tok:>8} {r_float['us_per_call']:>10} {r_float['ns_per_token']:>10}")
            speedup = r_float['us_per_call'] / max(r_i64['us_per_call'], 0.01)
            print(f"{'  -> speedup':<25} {'':>8} {f'{speedup:.0f}x':>10}")
            print()

        print("\n--- Full pipeline ---")
        for n_tok in [256, 1024, 4096]:
            r = bench_full_pipeline(n_tok, num_experts=args.num_experts)
            print(f"  {n_tok} tokens: {r['ms_per_call']} ms/call, {r['tokens_per_sec']:,} tok/s")

    elif args.mode == "engine":
        from benchmarks.bench_engine import run_full_benchmark
        run_full_benchmark(
            num_requests=args.requests,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            concurrency=args.concurrency,
            num_experts=args.num_experts,
        )

    else:
        # Default: run both
        from benchmarks.bench_engine import run_full_benchmark
        run_full_benchmark(
            num_requests=args.requests,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            concurrency=args.concurrency,
            num_experts=args.num_experts,
        )

    print("\nDone.")


def cmd_agent(args):
    """Run the agentic loop against a vllm-i64 server."""
    from vllm_i64.agentics import Agent

    agent = Agent(
        base_url=args.server,
        api_key=getattr(args, 'api_key', None),
        allow_shell=args.allow_shell,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=True,
    )

    # Check server is reachable
    if not agent.client.health():
        print(f"Error: cannot reach vllm-i64 server at {args.server}")
        print(f"Start it first: vllm-i64 serve pacific-prime-chat --checkpoint ... --port 8000")
        sys.exit(1)

    if args.interactive:
        agent.interactive()
    elif args.task:
        agent.run(" ".join(args.task))
    else:
        # No task given, start interactive
        agent.interactive()


def main():
    parser = argparse.ArgumentParser(
        prog="vllm-i64",
        description="Integer-first inference engine for token-routed models",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start inference server")
    p_serve.add_argument("model", help="Model name (e.g. pacific-prime-chat)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p_serve.add_argument("--tp", type=int, default=1, help="Tensor parallel size (num GPUs)")
    p_serve.add_argument("--pp", type=int, default=1, help="Pipeline parallel size (num stages)")
    p_serve.add_argument("--quantization", default="int8",
                         choices=["int8", "int4", "awq", "gptq", "none"],
                         help="Weight quantization (default: int8 — native integer compute; "
                              "awq/gptq for pre-quantized HF checkpoints)")
    p_serve.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    p_serve.add_argument("--chat-template", default=None, help="Path to chat template")
    p_serve.add_argument("--enable-prefix-caching", action="store_true",
                         help="Enable prefix caching for KV cache reuse")
    p_serve.add_argument("--kv-cache-dtype", default=None, choices=["fp8", "fp8_e5m2"],
                         help="KV cache quantization (fp8 for 2x memory savings)")
    p_serve.add_argument("--speculative-model", default=None,
                         help="Path to draft model for speculative decoding")
    p_serve.add_argument("--num-speculative-tokens", type=int, default=5,
                         help="Number of tokens to speculate ahead")
    p_serve.add_argument("--compile", action="store_true",
                         help="Enable torch.compile for kernel fusion (10-30%% speedup)")
    p_serve.add_argument("--compile-mode", default="reduce-overhead",
                         choices=["default", "reduce-overhead", "max-autotune"],
                         help="torch.compile mode (default: reduce-overhead)")
    p_serve.add_argument("--no-cuda-graphs", action="store_true",
                         help="Disable CUDA graph capture (saves ~4 GiB VRAM, use when GPU is shared)")
    p_serve.add_argument("--max-kv-blocks", type=int, default=0,
                         help="Total KV cache blocks (0 = auto: max(256, max_seqs*8)). "
                              "Reduce to save VRAM, e.g. --max-kv-blocks 128")
    p_serve.add_argument("--enable-swap", action="store_true",
                         help="Enable swap-to-CPU for KV cache overflow")
    p_serve.add_argument("--api-key", default=None,
                         help="API key for bearer token authentication")
    p_serve.add_argument("--rate-limit", type=int, default=0,
                         help="Max requests per minute per IP (0 = unlimited)")
    p_serve.add_argument("--max-pending", type=int, default=0,
                         help="Max pending requests before rejecting (0 = unlimited)")
    p_serve.add_argument("--rag-index", default=None,
                         help="Path to RAG index directory (enables /v1/rag/* endpoints)")
    p_serve.add_argument("--disaggregated", action="store_true",
                         help="Enable disaggregated prefill/decode (requires TP >= 2). "
                              "GPU 0 handles prefill (compute-bound), GPU 1 handles decode (memory-bound)")
    p_serve.set_defaults(func=cmd_serve)

    # list
    p_list = sub.add_parser("list", help="List registered models")
    p_list.set_defaults(func=cmd_list)

    # check
    p_check = sub.add_parser("check", help="Check model availability")
    p_check.add_argument("model", help="Model name")
    p_check.set_defaults(func=cmd_check)

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmarks")
    p_bench.add_argument("--mode", default="all", choices=["all", "routing", "engine"],
                         help="Benchmark mode: routing, engine, or all (default)")
    p_bench.add_argument("--num-experts", type=int, default=4)
    p_bench.add_argument("--requests", type=int, default=20, help="Number of requests")
    p_bench.add_argument("--prompt-len", type=int, default=64, help="Prompt length (tokens)")
    p_bench.add_argument("--output-len", type=int, default=64, help="Max output tokens")
    p_bench.add_argument("--concurrency", type=int, default=8, help="Async concurrency")
    p_bench.set_defaults(func=cmd_bench)

    # agent
    p_agent = sub.add_parser("agent", help="Run AI agent (connects to a running vllm-i64 server)")
    p_agent.add_argument("task", nargs="*", help="Task to execute (omit for interactive mode)")
    p_agent.add_argument("--server", default="http://localhost:8000",
                         help="vllm-i64 server URL (default: http://localhost:8000)")
    p_agent.add_argument("--interactive", "-i", action="store_true",
                         help="Interactive REPL mode")
    p_agent.add_argument("--allow-shell", action="store_true",
                         help="Enable shell command execution (dangerous)")
    p_agent.add_argument("--max-steps", type=int, default=20,
                         help="Max agent steps before stopping (default: 20)")
    p_agent.add_argument("--temperature", type=float, default=0.6,
                         help="LLM temperature (default: 0.6)")
    p_agent.add_argument("--max-tokens", type=int, default=1024,
                         help="Max tokens per LLM response (default: 1024)")
    p_agent.add_argument("--api-key", default=None,
                         help="API key for server authentication")
    p_agent.set_defaults(func=cmd_agent)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate --tp and --pp
    if hasattr(args, 'tp') and args.tp < 1:
        parser.error("--tp must be >= 1")
    if hasattr(args, 'pp') and args.pp < 1:
        parser.error("--pp must be >= 1")

    args.func(args)


if __name__ == "__main__":
    main()
