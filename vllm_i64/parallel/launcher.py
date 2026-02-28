"""
vllm-i64 :: Distributed Launcher

torchrun wrapper for multi-GPU tensor-parallel and pipeline-parallel inference.

Usage (CLI):
    vllm-i64 serve pacific-prime-chat --tp 4
    → internally launches:
    torchrun --nproc_per_node=4 -m vllm_i64.parallel.worker serve pacific-prime-chat

    vllm-i64 serve pacific-prime-chat --tp 2 --pp 2
    → launches 4 workers total (2 TP × 2 PP stages)

Usage (Python):
    launch_distributed(tp_size=4, args=["serve", "pacific-prime-chat"])
    launch_distributed(tp_size=2, pp_size=2, args=["serve", "pacific-prime-chat"])

INL - 2025
"""

import subprocess
import sys
import os


def launch_distributed(tp_size: int, args: list, pp_size: int = 1) -> int:
    """
    Launch vllm-i64 with torchrun for tensor and/or pipeline parallelism.

    Args:
        tp_size: number of GPUs per pipeline stage (tensor parallel degree)
        args: CLI arguments (e.g. ["serve", "pacific-prime-chat", "--port", "8000"])
        pp_size: number of pipeline stages (pipeline parallel degree)

    Returns:
        process return code
    """
    nproc = tp_size * pp_size

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port", _find_free_port(),
        "-m", "vllm_i64.parallel.worker",
    ] + args

    env = os.environ.copy()
    env["VLLM_I64_TP_SIZE"] = str(tp_size)
    env["VLLM_I64_PP_SIZE"] = str(pp_size)

    print(f"vllm-i64 :: launching {nproc} workers (TP={tp_size}, PP={pp_size})")
    print(f"  cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _find_free_port() -> str:
    """Find a free port for distributed master."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])
