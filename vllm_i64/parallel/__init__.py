"""
vllm-i64 :: Parallelism

Shard expert weights and attention heads across GPUs (TP).
Distribute layers across pipeline stages (PP).
Disaggregated prefill/decode across GPU groups (DP).
Token routing stays on all ranks (integer, cheap).
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

from vllm_i64.parallel.pipeline_parallel import (
    PPState,
    init_pp,
    get_pp,
    get_pp_rank,
    get_pp_world_size,
    is_first_pp_rank,
    is_last_pp_rank,
    get_pp_indices,
)

from vllm_i64.parallel.pp_utils import (
    PPMissingLayer,
    IntermediateTensors,
    make_layers,
)

from vllm_i64.parallel.disaggregated import (
    DisaggRole,
    KVTransfer,
    PrefillWorker,
    DecodeWorker,
    DisaggregatedCoordinator,
    setup_disaggregated,
    launch_disaggregated,
)

__all__ = [
    "TPState", "ColumnParallelLinear", "RowParallelLinear",
    "init_distributed", "get_tp", "get_tp_rank", "get_tp_world_size",
    "shard_expert_weights", "all_reduce",
    "PPState", "init_pp", "get_pp", "get_pp_rank", "get_pp_world_size",
    "is_first_pp_rank", "is_last_pp_rank", "get_pp_indices",
    "PPMissingLayer", "IntermediateTensors", "make_layers",
    "DisaggRole", "KVTransfer", "PrefillWorker", "DecodeWorker",
    "DisaggregatedCoordinator", "setup_disaggregated", "launch_disaggregated",
]
