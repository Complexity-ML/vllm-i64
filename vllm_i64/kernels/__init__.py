"""
vllm-i64 :: GPU Kernels

Three acceleration tiers (auto-selected, highest available wins):

  1. Custom CUDA (I64_*.cu) — maximum throughput, JIT compiled
     Fused RMSNorm+quant, INT8/FP8 GEMM, integer softmax LUT,
     fused RoPE, fused gate+up+SiLU

  2. Triton (@triton.jit) — near-CUDA performance, Python-authored
     Same fused ops as CUDA, easier to modify/tune

  3. PyTorch fallback — works everywhere (CPU, old GPU, no compiler)
     Standard torch ops, no compilation needed

INL - 2025
"""
