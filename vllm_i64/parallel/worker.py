"""
vllm-i64 :: Distributed Worker

Each worker process is launched by torchrun.
Initializes distributed (TP + PP), loads the model shard, and serves.

Only rank 0 runs the API server.
Other ranks participate in TP compute via all_reduce
and PP via send/recv.

INL - 2025
"""

import os
import sys
import torch


def main():
    """Worker entry point — called by torchrun."""
    from vllm_i64.parallel.tensor_parallel import init_distributed, get_tp
    from vllm_i64.parallel.pipeline_parallel import init_pp, get_pp

    tp_size = int(os.environ.get("VLLM_I64_TP_SIZE", "1"))
    pp_size = int(os.environ.get("VLLM_I64_PP_SIZE", "1"))

    init_distributed(tp_size=tp_size)
    init_pp(pp_size=pp_size)

    tp = get_tp()
    pp = get_pp()

    # Parse remaining args (same as CLI)
    args = sys.argv[1:]
    if not args:
        if tp.tp_rank == 0 and pp.pp_rank == 0:
            print("Usage: worker <serve|bench> [args...]")
        return

    command = args[0]

    if command == "serve":
        _run_serve(args[1:], tp, pp)
    elif command == "bench":
        _run_bench(args[1:], tp, pp)
    else:
        if tp.tp_rank == 0 and pp.pp_rank == 0:
            print(f"Unknown command: {command}")


def _run_serve(args: list, tp, pp):
    """Run the serve command in distributed mode."""
    import argparse
    from vllm_i64.core.loader import load_model_by_name
    from vllm_i64.engine.i64_engine import I64Engine
    from vllm_i64.core.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--quantization", default=None)
    parsed = parser.parse_args(args)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[parsed.dtype]

    if tp.tp_rank == 0 and pp.pp_rank == 0:
        print(f"vllm-i64 :: serving {parsed.model} (TP={tp.tp_size}, PP={pp.pp_size})")

    # Each rank loads its shard (TP shards weights, PP shards layers)
    model = load_model_by_name(
        parsed.model, dtype=dtype, device=tp.device,
        checkpoint_override=parsed.checkpoint,
        quantization=parsed.quantization,
    )
    model.eval()

    engine = I64Engine(
        model=model,
        num_experts=model.config.num_experts,
        vocab_size=model.config.vocab_size,
        device=tp.device,
    )

    if tp.tp_rank == 0 and pp.pp_rank == 0:
        # Only global rank 0 runs the API server
        from vllm_i64.api.server import I64Server

        tokenizer = load_tokenizer(parsed.model)
        chat_template = None
        if parsed.chat_template:
            with open(parsed.chat_template) as f:
                chat_template = f.read()

        server = I64Server(
            engine=engine,
            tokenizer=tokenizer,
            chat_template=chat_template,
            model_name=parsed.model,
            host=parsed.host,
            port=parsed.port,
        )
        server.run()
    else:
        # Non-rank-0 workers: participate in TP/PP compute
        _worker_loop(engine, tp, pp)


def _worker_loop(engine, tp, pp):
    """
    Non-rank-0 worker loop.

    Workers participate in TP compute by running the same forward passes
    as rank 0. Coordination protocol:

      1. Rank 0 broadcasts a control tensor: [opcode, batch_size, seq_len]
         - opcode 0: shutdown
         - opcode 1: forward step
      2. Workers receive control, then broadcast tensor data (token_ids, positions)
      3. Workers run the same model.forward() → NCCL all_reduce inside RowParallel
      4. PP workers send/recv intermediate tensors between stages
      5. Loop back to step 1

    This replaces the barrier-only approach with real data synchronization.
    """
    import torch.distributed as dist

    global_rank = tp.tp_rank + pp.pp_rank * tp.tp_size
    if global_rank == 0:
        return

    print(f"[Worker TP={tp.tp_rank} PP={pp.pp_rank}] ready, waiting for compute")

    device = tp.device

    while True:
        try:
            # 1. Receive control signal from rank 0
            control = torch.zeros(3, dtype=torch.int64, device=device)
            dist.broadcast(control, src=0, group=tp.tp_group)

            opcode = control[0].item()
            if opcode == 0:
                # Shutdown
                print(f"[Worker TP={tp.tp_rank} PP={pp.pp_rank}] shutdown signal received")
                break

            batch_tokens = control[1].item()

            # 2. Receive tensor data
            token_ids = torch.zeros(batch_tokens, dtype=torch.int64, device=device)
            positions = torch.zeros(batch_tokens, dtype=torch.int32, device=device)
            dist.broadcast(token_ids, src=0, group=tp.tp_group)
            dist.broadcast(positions, src=0, group=tp.tp_group)

            # 3. Run forward pass (participates in all_reduce via RowParallel)
            with torch.no_grad():
                engine.model(token_ids=token_ids, positions=positions)

        except Exception as e:
            print(f"[Worker TP={tp.tp_rank} PP={pp.pp_rank}] error: {e}")
            break

    print(f"[Worker TP={tp.tp_rank} PP={pp.pp_rank}] exiting")


def _run_bench(args: list, tp, pp):
    """Run benchmarks in distributed mode."""
    if tp.tp_rank == 0 and pp.pp_rank == 0:
        from vllm_i64.cli import cmd_bench
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-experts", type=int, default=4)
        parsed = parser.parse_args(args)
        cmd_bench(parsed)


if __name__ == "__main__":
    main()
