"""
vllm-i64 :: TP Worker

Each worker process is launched by torchrun.
Initializes distributed, loads the model shard, and serves.

Only rank 0 runs the API server.
Other ranks participate in TP compute via all_reduce.

INL - 2025
"""

import os
import sys
import torch


def main():
    """Worker entry point — called by torchrun."""
    from vllm_i64.parallel.tensor_parallel import init_distributed, get_tp

    tp_size = int(os.environ.get("VLLM_I64_TP_SIZE", "1"))
    init_distributed(tp_size=tp_size)

    tp = get_tp()

    # Parse remaining args (same as CLI)
    args = sys.argv[1:]
    if not args:
        if tp.tp_rank == 0:
            print("Usage: worker <serve|bench> [args...]")
        return

    command = args[0]

    if command == "serve":
        _run_serve(args[1:], tp)
    elif command == "bench":
        _run_bench(args[1:], tp)
    else:
        if tp.tp_rank == 0:
            print(f"Unknown command: {command}")


def _run_serve(args: list, tp):
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
    parsed = parser.parse_args(args)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[parsed.dtype]

    if tp.tp_rank == 0:
        print(f"vllm-i64 :: serving {parsed.model} (TP={tp.tp_size})")

    # Each rank loads its shard
    model = load_model_by_name(
        parsed.model, dtype=dtype, device=tp.device,
        checkpoint_override=parsed.checkpoint,
    )
    model.eval()

    engine = I64Engine(
        model=model,
        num_experts=4,
        vocab_size=model.config.vocab_size,
        device=tp.device,
    )

    if tp.tp_rank == 0:
        # Only rank 0 runs the API server
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
        # Non-rank-0 workers: participate in TP compute
        # They wait for all_reduce calls from rank 0's forward passes
        _worker_loop(engine, tp)


def _worker_loop(engine, tp):
    """
    Non-rank-0 worker loop.

    Workers participate in TP compute by running the same forward passes
    as rank 0. Coordination protocol:

      1. Rank 0 broadcasts a control tensor: [opcode, batch_size, seq_len]
         - opcode 0: shutdown
         - opcode 1: forward step
      2. Workers receive control, then broadcast tensor data (token_ids, positions)
      3. Workers run the same model.forward() → NCCL all_reduce inside RowParallel
      4. Loop back to step 1

    This replaces the barrier-only approach with real data synchronization.
    """
    import torch.distributed as dist

    if tp.tp_rank == 0:
        return

    print(f"[TP worker {tp.tp_rank}] ready, waiting for compute")

    device = tp.device

    while True:
        try:
            # 1. Receive control signal from rank 0
            control = torch.zeros(3, dtype=torch.int64, device=device)
            dist.broadcast(control, src=0, group=tp.tp_group)

            opcode = control[0].item()
            if opcode == 0:
                # Shutdown
                print(f"[TP worker {tp.tp_rank}] shutdown signal received")
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
            print(f"[TP worker {tp.tp_rank}] error: {e}")
            break

    print(f"[TP worker {tp.tp_rank}] exiting")


def _run_bench(args: list, tp):
    """Run benchmarks in distributed mode."""
    if tp.tp_rank == 0:
        from vllm_i64.cli import cmd_bench
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-experts", type=int, default=4)
        parsed = parser.parse_args(args)
        cmd_bench(parsed)


if __name__ == "__main__":
    main()
