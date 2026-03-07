"""
vllm-i64 :: I64 CUDA Kernel Loader

JIT compiles all I64_*.cu kernels into a single PyTorch extension.
Falls back gracefully when CUDA toolkit or GPU is not available.

Usage:
    from vllm_i64.kernels.cuda import get_i64_cuda_ops, is_i64_cuda_available

    if is_i64_cuda_available():
        ops = get_i64_cuda_ops()
        out = ops.rmsnorm_forward(x, weight, eps)
        x_int8, scale = ops.quantize_int8(x)
        y = ops.gemm_dequant_int8(x, w_int8, w_scale)

INL - 2025
"""

import os
import torch
from typing import Optional
from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.kernels.cuda")

_I64_OPS = None
_I64_COMPILE_ATTEMPTED = False


def _get_csrc_dir() -> str:
    """Get path to csrc/ directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))),
        "csrc",
    )


def _try_compile_i64() -> Optional[object]:
    """JIT compile all I64_*.cu kernels."""
    global _I64_OPS, _I64_COMPILE_ATTEMPTED

    if _I64_COMPILE_ATTEMPTED:
        return _I64_OPS
    _I64_COMPILE_ATTEMPTED = True

    if not torch.cuda.is_available():
        logger.info("CUDA not available, I64 GPU kernels disabled")
        return None

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = _get_csrc_dir()

        # All I64_*.cu source files
        sources = [
            os.path.join(csrc_dir, "I64_binding.cu"),
            os.path.join(csrc_dir, "I64_rmsnorm.cu"),
            os.path.join(csrc_dir, "I64_rope.cu"),
            os.path.join(csrc_dir, "I64_quantize.cu"),
            os.path.join(csrc_dir, "I64_softmax.cu"),
            os.path.join(csrc_dir, "I64_gemm.cu"),
        ]

        # Check all sources exist
        for src in sources:
            if not os.path.exists(src):
                logger.warning("I64 CUDA source not found: %s", src)
                return None

        logger.info("Compiling I64 CUDA kernels (first time may take a minute)...")

        _I64_OPS = load(
            name="vllm_i64_gpu_kernels",
            sources=sources,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",              # debug info without perf cost
            ],
            verbose=False,
        )

        # Initialize LUT in constant memory
        _I64_OPS.init_softmax_lut()

        logger.info("I64 CUDA kernels compiled successfully")
        return _I64_OPS

    except Exception as e:
        logger.warning("I64 CUDA kernel compilation failed: %s", e)
        logger.info("Falling back to Triton/PyTorch kernels")
        return None


def get_i64_cuda_ops():
    """
    Get compiled I64 CUDA ops, or None.

    Available ops (when compiled):
        rmsnorm_forward(x, weight, eps) → out
        rmsnorm_quant_forward(x, weight, eps) → (out_int8, scale)
        rope_forward(x, cos, sin) → out
        rope_integer_forward(x, cos_q14, sin_q14) → out
        quantize_int8(x) → (x_int8, scale)
        dequantize_int8(x_int8, scale) → out
        quantize_perchannel_int8(weight) → (w_int8, scale)
        softmax_integer(logits) → out
        gemm_dequant_int8(x, w_int8, w_scale) → out
        gemm_silu_int8(x, gate_int8, gate_scale, up_int8, up_scale) → out
    """
    return _try_compile_i64()


def is_i64_cuda_available() -> bool:
    """Check if compiled I64 CUDA kernels are available."""
    return _try_compile_i64() is not None
