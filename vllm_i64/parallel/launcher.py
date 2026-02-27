"""
vllm-i64 :: Distributed Launcher

torchrun wrapper for multi-GPU tensor-parallel inference.

Usage (CLI):
    vllm-i64 serve pacific-prime-chat --tp 4
    → internally launches:
    torchrun --nproc_per_node=4 -m vllm_i64.parallel.worker serve pacific-prime-chat

Usage (Python):
    launch_distributed(tp_size=4, args=["serve", "pacific-prime-chat"])

INL - 2025
"""

import subprocess
import sys
import os


def launch_distributed(tp_size: int, args: list) -> int:
    """
    Launch vllm-i64 with torchrun for tensor parallelism.

    Args:
        tp_size: number of GPUs (tensor parallel degree)
        args: CLI arguments (e.g. ["serve", "pacific-prime-chat", "--port", "8000"])

    Returns:
        process return code
    """
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc_per_node={tp_size}",
        "--master_port", _find_free_port(),
        "-m", "vllm_i64.parallel.worker",
    ] + args

    env = os.environ.copy()
    env["VLLM_I64_TP_SIZE"] = str(tp_size)

    print(f"vllm-i64 :: launching {tp_size} workers")
    print(f"  cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _find_free_port() -> str:
    """Find a free port for distributed master."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])
