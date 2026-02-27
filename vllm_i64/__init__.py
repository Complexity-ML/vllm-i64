"""
vllm-i64: Integer-first inference engine for token-routed models.

The philosophy: integers everywhere, float only where math requires it.

  Routing:     i64 (token_id & mask)
  Scheduling:  i64 (counters, block tables, request IDs)
  KV cache:    i32 (block indices)
  Sampling:    i64 (argmax → token ID)
  Compute:     fp16 (expert MLP, attention — the ONLY float)

INL - 2025
"""

__version__ = "0.1.0"
