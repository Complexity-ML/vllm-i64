"""
vllm-i64 :: Tensor Parallelism

Shard expert weights and attention heads across GPUs.
Token routing stays on all ranks (integer, cheap).
Expert compute + attention is distributed.
"""

from vllm_i64.parallel.tensor_parallel import (
    TPState,
    ColumnParallelLinear,
    RowParallelLinear,
    init_distributed,
    get_tp,
    get_tp_rank,
    get_tp_world_size,
    shard_expert_weights,
    all_reduce,
)
