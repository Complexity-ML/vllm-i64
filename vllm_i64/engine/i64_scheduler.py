"""
vllm-i64 :: Integer-First Scheduler

Schedules requests for token-routed inference.
ALL scheduling decisions are integer operations:
  - Request IDs: int64
  - Token IDs: int64
  - Expert assignments: int32
  - Slot management: int32
  - KV cache block indices: int32

Zero float in the scheduler. FP16 exists only in model forward pass.

INL - 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum


class RequestStatus(IntEnum):
    """Request lifecycle — all integer states."""
    PENDING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED = 3


@dataclass
class I64Request:
    """
    A single inference request.
    All fields are integer or integer arrays — zero float.
    """
    request_id: int                           # i64
    prompt_token_ids: np.ndarray              # [prompt_len] i64
    max_new_tokens: int                       # i32
    status: RequestStatus = RequestStatus.PENDING

    # Generated tokens (filled during inference)
    output_token_ids: List[int] = field(default_factory=list)

    # KV cache block indices (integer)
    kv_block_ids: List[int] = field(default_factory=list)

    # Position in sequence (integer counter)
    seq_pos: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generated

    @property
    def is_finished(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    def get_all_token_ids(self) -> np.ndarray:
        """All tokens (prompt + generated) as i64 array."""
        all_ids = list(self.prompt_token_ids) + self.output_token_ids
        return np.array(all_ids, dtype=np.int64)

    def get_last_token_id(self) -> int:
        """Last token ID (for autoregressive step)."""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return int(self.prompt_token_ids[-1])


@dataclass
class I64Batch:
    """
    A batch of requests scheduled for one forward pass.
    All indexing is integer.
    """
    request_ids: np.ndarray          # [batch_size] i64
    token_ids: np.ndarray            # [total_tokens] i64
    expert_ids: np.ndarray           # [total_tokens] i32 — pre-computed routing
    seq_lens: np.ndarray             # [batch_size] i32
    positions: np.ndarray            # [total_tokens] i32 — position in sequence
    kv_block_table: np.ndarray       # [batch_size, max_blocks] i32

    # Prefill vs decode flags (integer)
    is_prefill: np.ndarray           # [batch_size] i32 (0 or 1)

    # Tokens per request in this batch (for KV cache routing)
    tokens_per_request: np.ndarray = None  # [batch_size] i32

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)


class I64Scheduler:
    """
    Integer-first scheduler for token-routed models.

    Key design: ALL scheduling decisions are integer operations.
    - Token routing: token_id & expert_mask (i64)
    - Slot allocation: integer counters
    - KV cache management: integer block table
    - Priority: FCFS with integer timestamps

    No float anywhere in the scheduler.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        num_experts: int = 4,
        kv_block_size: int = 16,
        max_kv_blocks: int = 4096,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.expert_mask = np.int64(num_experts - 1)  # For bit masking
        self.kv_block_size = kv_block_size

        # Request queues (integer indexed)
        self.pending: List[I64Request] = []
        self.running: List[I64Request] = []
        self.finished: List[I64Request] = []

        # KV cache block allocator (integer)
        self.free_blocks = list(range(max_kv_blocks))
        self.max_kv_blocks = max_kv_blocks

        # Integer counters
        self.next_request_id: int = 0
        self.step_counter: int = 0

    def add_request(self, prompt_token_ids: np.ndarray, max_new_tokens: int = 256) -> int:
        """
        Add a new request. Returns integer request_id.
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        req = I64Request(
            request_id=request_id,
            prompt_token_ids=np.asarray(prompt_token_ids, dtype=np.int64),
            max_new_tokens=max_new_tokens,
        )
        self.pending.append(req)
        return request_id

    def _allocate_kv_blocks(self, num_blocks: int) -> Optional[List[int]]:
        """Allocate KV cache blocks. Pure integer."""
        if len(self.free_blocks) < num_blocks:
            return None
        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return allocated

    def _free_kv_blocks(self, block_ids: List[int]):
        """Free KV cache blocks. Pure integer."""
        self.free_blocks.extend(block_ids)

    def _compute_expert_ids(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Pre-compute expert routing for all tokens in batch.
        Pure i64 bit masking — zero float.
        """
        return (token_ids & self.expert_mask).astype(np.int32)

    def schedule(self) -> Optional[I64Batch]:
        """
        Schedule the next batch. All integer decisions.

        Continuous batching: mix prefill and decode requests.
        """
        self.step_counter += 1

        # Move finished requests out
        still_running = []
        for req in self.running:
            if req.is_finished:
                req.status = RequestStatus.FINISHED
                self._free_kv_blocks(req.kv_block_ids)
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

        # Try to admit new requests
        while self.pending and len(self.running) < self.max_batch_size:
            req = self.pending[0]
            num_blocks_needed = (req.num_prompt_tokens + self.kv_block_size - 1) // self.kv_block_size
            blocks = self._allocate_kv_blocks(num_blocks_needed)
            if blocks is None:
                break  # No KV cache space
            req.kv_block_ids = blocks
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            self.pending.pop(0)

        if not self.running:
            return None

        # Build batch — all integer arrays
        request_ids = []
        all_token_ids = []
        all_positions = []
        seq_lens = []
        is_prefill = []
        toks_per_req = []
        max_blocks_per_seq = 0

        for req in self.running:
            request_ids.append(req.request_id)

            if req.seq_pos == 0:
                # Prefill: process all prompt tokens
                tokens = req.prompt_token_ids
                positions = np.arange(len(tokens), dtype=np.int32)
                is_prefill.append(1)
            else:
                # Decode: process last token only
                tokens = np.array([req.get_last_token_id()], dtype=np.int64)
                positions = np.array([req.total_tokens - 1], dtype=np.int32)
                is_prefill.append(0)

            all_token_ids.append(tokens)
            all_positions.append(positions)
            seq_lens.append(req.total_tokens)
            toks_per_req.append(len(tokens))
            max_blocks_per_seq = max(max_blocks_per_seq, len(req.kv_block_ids))

        # Concatenate (integer arrays)
        token_ids_flat = np.concatenate(all_token_ids).astype(np.int64)
        positions_flat = np.concatenate(all_positions).astype(np.int32)

        # Pre-compute routing — pure i64
        expert_ids = self._compute_expert_ids(token_ids_flat)

        # KV block table (padded, integer)
        kv_block_table = np.zeros(
            (len(self.running), max(max_blocks_per_seq, 1)),
            dtype=np.int32
        )
        for i, req in enumerate(self.running):
            for j, block_id in enumerate(req.kv_block_ids):
                kv_block_table[i, j] = block_id

        return I64Batch(
            request_ids=np.array(request_ids, dtype=np.int64),
            token_ids=token_ids_flat,
            expert_ids=expert_ids,
            seq_lens=np.array(seq_lens, dtype=np.int32),
            positions=positions_flat,
            kv_block_table=kv_block_table,
            is_prefill=np.array(is_prefill, dtype=np.int32),
            tokens_per_request=np.array(toks_per_req, dtype=np.int32),
        )

    def update_after_step(self, new_token_ids: Dict[int, int]):
        """
        Update requests after a forward step.

        new_token_ids: {request_id: generated_token_id} — all integers.
        """
        for req in self.running:
            if req.request_id in new_token_ids:
                token_id = new_token_ids[req.request_id]
                req.output_token_ids.append(token_id)
                req.seq_pos = req.total_tokens

                # Allocate more KV blocks if needed (integer arithmetic)
                blocks_needed = (req.total_tokens + self.kv_block_size - 1) // self.kv_block_size
                while len(req.kv_block_ids) < blocks_needed:
                    new_blocks = self._allocate_kv_blocks(1)
                    if new_blocks:
                        req.kv_block_ids.extend(new_blocks)
                    else:
                        break

    def get_stats(self) -> Dict[str, int]:
        """Stats — all integer values."""
        return {
            "pending": len(self.pending),
            "running": len(self.running),
            "finished": len(self.finished),
            "free_kv_blocks": len(self.free_blocks),
            "total_steps": self.step_counter,
        }
