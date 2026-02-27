"""
vllm-i64 :: Kernel Loader

JIT compiles and loads CUDA kernels from csrc/.
Falls back to PyTorch implementations in i64_ops.py when:
  - CUDA is not available
  - Compilation fails
  - Running on CPU

Usage:
    from vllm_i64.kernels.kernel_loader import get_ops

    ops = get_ops()
    expert_ids = ops.route_tokens(token_ids, num_experts)
    intermediate = ops.silu_hadamard(gate_up, expert_inter)

INL - 2025
"""

import os
import torch
from typing import Optional
from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.kernels")

_COMPILED_OPS = None
_COMPILE_ATTEMPTED = False


class FallbackOps:
    """Pure PyTorch fallback implementations (from i64_ops.py)."""

    @staticmethod
    def route_tokens(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        expert_mask = num_experts - 1
        return (token_ids & expert_mask).to(torch.int32)

    @staticmethod
    def silu_hadamard(gate_up: torch.Tensor, expert_inter: int) -> torch.Tensor:
        gate = gate_up[:, :expert_inter]
        up = gate_up[:, expert_inter:]
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def scatter_by_expert(hidden, expert_ids, num_experts):
        from vllm_i64.kernels.i64_ops import i64_scatter
        return i64_scatter(hidden, expert_ids, num_experts)

    @staticmethod
    def gather_by_expert(expert_out, scatter_indices):
        from vllm_i64.kernels.i64_ops import i64_gather
        return i64_gather(expert_out, scatter_indices)


def _try_compile() -> Optional[object]:
    """Try to JIT compile CUDA kernels."""
    global _COMPILED_OPS, _COMPILE_ATTEMPTED

    if _COMPILE_ATTEMPTED:
        return _COMPILED_OPS
    _COMPILE_ATTEMPTED = True

    if not torch.cuda.is_available():
        logger.info("CUDA not available, using PyTorch fallback kernels")
        return None

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "csrc",
        )
        binding_path = os.path.join(csrc_dir, "i64_ops_binding.cu")

        if not os.path.exists(binding_path):
            logger.warning(f"CUDA source not found: {binding_path}")
            return None

        logger.info("Compiling CUDA kernels (first time may take a minute)...")

        _COMPILED_OPS = load(
            name="vllm_i64_kernels",
            sources=[binding_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )

        logger.info("CUDA kernels compiled successfully")
        return _COMPILED_OPS

    except Exception as e:
        logger.warning(f"CUDA kernel compilation failed: {e}")
        logger.info("Using PyTorch fallback kernels")
        return None


def get_ops():
    """
    Get the best available ops implementation.

    Returns compiled CUDA ops if available, otherwise PyTorch fallback.
    """
    compiled = _try_compile()
    if compiled is not None:
        return compiled
    return FallbackOps()


def is_cuda_kernels_available() -> bool:
    """Check if compiled CUDA kernels are available."""
    return _try_compile() is not None
