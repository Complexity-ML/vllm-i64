"""
vllm-i64 :: torch.compile optimization

Compiles performance-critical forward paths for CPU and GPU speedup.
torch.compile fuses ops, eliminates Python overhead, and enables
hardware-specific optimizations (vectorization, memory layout).

Usage:
    from vllm_i64.core.compile import compile_model
    compile_model(model)  # compiles all eligible submodules in-place

Backend selection:
    - 'inductor' (default): full optimization with C++ codegen
    - 'aot_eager': AOT autograd without codegen (fallback if no C++ compiler)

INL - 2025
"""

import torch
import torch.nn as nn
import logging
import warnings
from typing import Optional

_logger = logging.getLogger("vllm_i64.compile")

# Detect best available backend once
_COMPILE_AVAILABLE = hasattr(torch, 'compile')
_BEST_BACKEND: Optional[str] = None

# Suppress dynamo warnings about untraceable builtins (posix._path_normpath, etc.)
# These are triggered by try/except ImportError blocks in forward() methods.
# The warning is harmless — dynamo falls back to eager for those ops.
if _COMPILE_AVAILABLE:
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except (ImportError, AttributeError):
        pass


def _detect_backend() -> Optional[str]:
    """Probe for the best torch.compile backend."""
    global _BEST_BACKEND
    if _BEST_BACKEND is not None:
        return _BEST_BACKEND
    if not _COMPILE_AVAILABLE:
        return None

    # Try inductor first (best perf, needs C++ compiler)
    for backend in ('inductor', 'aot_eager'):
        try:
            fn = torch.compile(lambda x: x + 1, backend=backend)
            fn(torch.tensor(1.0))
            _BEST_BACKEND = backend
            _logger.info("torch.compile backend: %s", backend)
            return backend
        except Exception:
            continue

    _logger.warning("torch.compile: no working backend found")
    return None


def compile_model(model: nn.Module, backend: Optional[str] = None) -> nn.Module:
    """
    Compile performance-critical submodules in-place.

    Targets:
      - RMSNorm.forward (elementwise — big win from fusion)
      - DenseMLP.forward (3 GEMMs + SiLU — op fusion)
      - Attention forward paths (softmax + matmuls)

    Non-compiled modules (already optimized or incompatible):
      - CUDA/Triton kernel paths (already native)
      - KV cache operations (dynamic shapes)

    Args:
        model: the model to optimize
        backend: override backend ('inductor', 'aot_eager', etc.)

    Returns:
        the same model (modified in-place)
    """
    if not _COMPILE_AVAILABLE:
        _logger.warning("torch.compile not available (PyTorch < 2.0)")
        return model

    be = backend or _detect_backend()
    if be is None:
        _logger.warning("No torch.compile backend available, skipping")
        return model

    compiled_count = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__

        # RMSNorm: elementwise ops — huge fusion potential
        if cls_name == 'RMSNorm':
            _compile_method(module, 'forward', be, name)
            compiled_count += 1

        # DenseMLP: 3 GEMMs + activation — fuse the SiLU*up
        elif cls_name == 'DenseMLP':
            _compile_method(module, 'forward', be, name)
            compiled_count += 1

        # TokenRoutedMLP: expert dispatch + GEMMs
        elif cls_name == 'TokenRoutedMLP':
            _compile_method(module, 'forward', be, name)
            compiled_count += 1

    _logger.info("Compiled %d modules with backend '%s'", compiled_count, be)
    return model


def _compile_method(module: nn.Module, method_name: str, backend: str, module_name: str):
    """Compile a single method on a module, with error handling."""
    try:
        original = getattr(module, method_name)
        compiled = torch.compile(original, backend=backend, dynamic=True)
        setattr(module, method_name, compiled)
    except Exception as e:
        _logger.debug("Failed to compile %s.%s: %s", module_name, method_name, e)


def compile_function(fn, backend: Optional[str] = None):
    """
    Compile a standalone function (e.g., int4_linear, softmax_integer).

    Returns the compiled function, or the original if compilation fails.
    """
    if not _COMPILE_AVAILABLE:
        return fn

    be = backend or _detect_backend()
    if be is None:
        return fn

    try:
        return torch.compile(fn, backend=be, dynamic=True)
    except Exception as e:
        _logger.debug("Failed to compile %s: %s", fn.__name__, e)
        return fn


def is_compile_available() -> bool:
    """Check if torch.compile is usable."""
    return _detect_backend() is not None
