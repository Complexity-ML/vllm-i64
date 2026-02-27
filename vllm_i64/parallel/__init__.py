"""
vllm-i64 :: Tensor Parallelism

Shard expert weights across GPUs.
Token routing stays on all ranks (integer, cheap).
Expert compute is distributed.
"""

from vllm_i64.parallel.tensor_parallel import (
    TPConfig,
    shard_expert_weights,
    all_reduce_output,
    get_tp_rank,
    get_tp_world_size,
)
