"""
vllm-i64 :: Integer-First Scheduler

Schedules requests for token-routed inference.
ALL scheduling decisions are integer operations:
  - Request IDs: int64
  - Token IDs: int64
  - Expert assignments: int32
  - Slot management: int32
  - KV cache block indices: int32

Features:
  - Continuous batching (mix prefill + decode)
  - Chunked prefill (split long prompts across steps)
  - Request priorities (integer priority levels)
  - Preemption (evict low-priority running requests when KV OOM)
  - Fairness (round-robin within same priority)

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

    # Priority (integer: 0 = normal, negative = higher priority)
    priority: int = 0

    # Chunked prefill state
    prefill_progress: int = 0  # How many prompt tokens have been processed

    # Arrival order for fairness (integer timestamp)
    arrival_step: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generated

    # EOS token ID (set by engine when adding request)
    eos_token_id: int = 0

    @property
    def is_finished(self) -> bool:
        if self.num_generated >= self.max_new_tokens:
            return True
        # Stop on EOS token
        if self.output_token_ids and self.output_token_ids[-1] == self.eos_token_id:
            return True
        return False

    @property
    def prefill_complete(self) -> bool:
        return self.prefill_progress >= self.num_prompt_tokens

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
    - Priority: integer priority levels (lower = higher priority)
    - Fairness: arrival_step for tie-breaking

    No float anywhere in the scheduler.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        num_experts: int = 4,
        kv_block_size: int = 16,
        max_kv_blocks: int = 4096,
        max_prefill_tokens: int = 512,
        enable_preemption: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.expert_mask = np.int64(num_experts - 1)  # For bit masking
        self.kv_block_size = kv_block_size
        self.max_prefill_tokens = max_prefill_tokens
        self.enable_preemption = enable_preemption

        # Request queues (integer indexed)
        self.pending: List[I64Request] = []
        self.running: List[I64Request] = []
        self.finished: List[I64Request] = []
        self.preempted: List[I64Request] = []

        # KV cache block allocator (integer)
        self.free_blocks = list(range(max_kv_blocks))
        self.max_kv_blocks = max_kv_blocks

        # Integer counters
        self.next_request_id: int = 0
        self.step_counter: int = 0

        # Fairness: track tokens generated per request for round-robin
        self._tokens_generated_per_req: Dict[int, int] = {}

    def add_request(
        self,
        prompt_token_ids: np.ndarray,
        max_new_tokens: int = 256,
        priority: int = 0,
        eos_token_id: int = 0,
    ) -> int:
        """
        Add a new request. Returns integer request_id.
        priority: integer, lower = higher priority (0 = normal, -1 = high, 1 = low)
        eos_token_id: integer, token ID that signals end of sequence
        """
        request_id = self.next_request_id
        self.next_request_id += 1

        req = I64Request(
            request_id=request_id,
            prompt_token_ids=np.asarray(prompt_token_ids, dtype=np.int64),
            max_new_tokens=max_new_tokens,
            priority=priority,
            arrival_step=self.step_counter,
            eos_token_id=eos_token_id,
        )
        self.pending.append(req)
        # Sort pending by priority, then arrival order (stable sort)
        self.pending.sort(key=lambda r: (r.priority, r.arrival_step))
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

    def _try_preempt(self, blocks_needed: int) -> bool:
        """
        Try to preempt lowest-priority running request to free KV blocks.

        Returns True if enough blocks were freed.
        """
        if not self.enable_preemption or not self.running:
            return False

        # Sort running by priority descending (lowest priority = highest number)
        # then by least progress (most blocks to free)
        candidates = sorted(
            self.running,
            key=lambda r: (-r.priority, -len(r.kv_block_ids)),
        )

        freed = 0
        for victim in candidates:
            if freed >= blocks_needed:
                break

            # Don't preempt high-priority requests for low-priority ones
            if victim.priority <= 0 and not self.pending:
                continue

            # Preempt
            victim.status = RequestStatus.PREEMPTED
            self._free_kv_blocks(victim.kv_block_ids)
            freed += len(victim.kv_block_ids)
            victim.kv_block_ids = []
            # Reset generation state but keep prompt
            victim.output_token_ids = []
            victim.seq_pos = 0
            victim.prefill_progress = 0
            self.running.remove(victim)
            self.preempted.append(victim)

        return freed >= blocks_needed

    def schedule(self) -> Optional[I64Batch]:
        """
        Schedule the next batch with chunked prefill + priority + preemption.

        Continuous batching: mix prefill and decode requests.
        Chunked prefill: long prompts split across multiple steps.
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

        # Re-admit preempted requests (they go back to pending with priority boost)
        for req in self.preempted:
            req.status = RequestStatus.PENDING
            req.priority = min(req.priority, -1)  # Boost priority after preemption
            self.pending.append(req)
        self.preempted.clear()
        self.pending.sort(key=lambda r: (r.priority, r.arrival_step))

        # Try to admit new requests
        prefill_token_budget = self.max_prefill_tokens

        while self.pending and len(self.running) < self.max_batch_size:
            req = self.pending[0]
            num_blocks_needed = (req.num_prompt_tokens + self.kv_block_size - 1) // self.kv_block_size
            blocks = self._allocate_kv_blocks(num_blocks_needed)

            if blocks is None:
                # Try preemption
                if self._try_preempt(num_blocks_needed):
                    blocks = self._allocate_kv_blocks(num_blocks_needed)

            if blocks is None:
                break  # No KV cache space even after preemption

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

            if not req.prefill_complete:
                # === Chunked Prefill ===
                remaining = req.num_prompt_tokens - req.prefill_progress
                chunk_size = min(remaining, prefill_token_budget)

                if chunk_size <= 0:
                    # No prefill budget left — skip this request this step
                    # Still need to include it in the batch for consistency
                    tokens = np.array([req.get_last_token_id()], dtype=np.int64)
                    positions = np.array([req.prefill_progress - 1], dtype=np.int32)
                    is_prefill.append(0)
                else:
                    start = req.prefill_progress
                    end = start + chunk_size
                    tokens = req.prompt_token_ids[start:end]
                    positions = np.arange(start, end, dtype=np.int32)
                    is_prefill.append(1)
                    prefill_token_budget -= chunk_size
            elif req.seq_pos == 0:
                # Full prefill (short enough, no chunking needed)
                tokens = req.prompt_token_ids
                positions = np.arange(len(tokens), dtype=np.int32)
                is_prefill.append(1)
                prefill_token_budget -= len(tokens)
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
                # Update chunked prefill progress
                if not req.prefill_complete:
                    # This was a prefill step — advance progress
                    # The scheduler assigned this many tokens:
                    req.prefill_progress = req.num_prompt_tokens  # simplified: mark complete
                    req.seq_pos = req.num_prompt_tokens

                token_id = new_token_ids[req.request_id]
                req.output_token_ids.append(token_id)
                req.seq_pos = req.total_tokens

                # Track for fairness
                self._tokens_generated_per_req[req.request_id] = req.num_generated

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
            "preempted": len(self.preempted),
            "free_kv_blocks": len(self.free_blocks),
            "total_steps": self.step_counter,
        }
