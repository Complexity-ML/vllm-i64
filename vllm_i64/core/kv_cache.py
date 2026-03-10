"""
vllm-i64 :: Paged KV Cache

Block-based KV cache — physical tensors + vLLM-style BlockPool allocator.

Architecture (mirroring vLLM v1):
  BlockPool   → allocation / LRU eviction / prefix caching  (block_pool.py)
  KV tensors  → k_cache / v_cache per layer  (this file)
  block_table → seq_id × block_idx → physical_block_id  (this file)

Same public API as before; swap-to-CPU and CUDA graph support preserved.

INL - 2025
"""

import heapq
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm_i64.core.block_pool import BlockPool, KVCacheBlock

logger = logging.getLogger(__name__)


class PagedKVCache:
    """
    Paged KV cache.

    Memory layout (flash_attn-compatible):
        k_cache / v_cache : (num_blocks, block_size, num_kv_heads, head_dim)
        block_table       : (max_seqs, max_blocks_per_seq) int32
        seq_lens          : (max_seqs,) int32

    Block allocation is handled by BlockPool (vLLM-style doubly-linked free
    list + ref counting + prefix caching).

    Sequence-level LRU eviction: when the pool is full, the least recently
    used sequence is freed to make room — no RuntimeError, no dropped requests.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        max_seqs: int = 64,
        max_blocks_per_seq: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        kv_cache_dtype: Optional[str] = None,
        enable_lru_eviction: bool = True,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_seqs = max_seqs
        self.compute_dtype = dtype
        self.enable_lru_eviction = enable_lru_eviction

        self.device = self._resolve_device(device)

        # KV storage dtype (FP8 for 2× memory savings when requested)
        if kv_cache_dtype == "fp8" and self.device != "cpu":
            self.kv_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2" and self.device != "cpu":
            self.kv_dtype = torch.float8_e5m2
        else:
            self.kv_dtype = dtype
        self.dtype = dtype  # backward compat — read_kv returns compute dtype

        self.max_blocks_per_seq = (
            max_blocks_per_seq if max_blocks_per_seq is not None else num_blocks
        )

        # ── Physical KV tensors ────────────────────────────────────────────
        # (num_blocks, block_size, num_kv_heads, head_dim) — flash_attn layout
        self.k_caches = [
            torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=self.kv_dtype, device=self.device,
            )
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=self.kv_dtype, device=self.device,
            )
            for _ in range(num_layers)
        ]

        # ── Block table ────────────────────────────────────────────────────
        # (max_seqs, max_blocks_per_seq) int32  — on GPU for hot-path reads
        self.block_table = torch.full(
            (max_seqs, self.max_blocks_per_seq), -1,
            dtype=torch.int32, device=self.device,
        )

        # ── Sequence lengths ───────────────────────────────────────────────
        self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32, device=self.device)

        # ── vLLM-style BlockPool ───────────────────────────────────────────
        # Manages free list (O(1) doubly-linked LRU) + ref counting +
        # prefix caching.  Prefix caching is opt-in via enable_prefix_caching().
        self.pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=False,   # enabled via enable_prefix_caching()
            block_size=block_size,
        )

        # seq_id → list of KVCacheBlock objects (ordered by block_idx)
        self._seq_blocks: Dict[int, List[KVCacheBlock]] = {}

        # Pinned sequences (never evicted during active decode)
        self._pinned_seq_ids: Set[int] = set()

        # ── Sequence-level LRU ─────────────────────────────────────────────
        # Pool evicts at block level; we additionally evict at sequence level
        # so the scheduler gets clean per-seq eviction notifications.
        self._access_counter: int = 0
        self._last_access: Dict[int, int] = {}
        self._lru_heap: List[Tuple[int, int]] = []  # (tick, seq_id) min-heap
        self._eviction_count: int = 0
        self._evicted_seq_ids: List[int] = []

        # ── Lazy block zeroing ─────────────────────────────────────────────
        # Zero GPU memory at allocation time (not at free time) so free_sequence
        # is O(1).  Blocks that need zeroing before next use are tracked here.
        self._dirty_blocks: Set[int] = set()

        # ── Swap-to-CPU ────────────────────────────────────────────────────
        self._swap_enabled = False
        self._cpu_k_caches: Optional[List[torch.Tensor]] = None
        self._cpu_v_caches: Optional[List[torch.Tensor]] = None
        self._cpu_free_blocks: List[int] = []
        self._cpu_block_table: Optional[torch.Tensor] = None
        self._swapped_seqs: Dict[int, dict] = {}
        self._swap_count: int = 0

        # ── CUDA graph support ─────────────────────────────────────────────
        self._graph_mode: bool = False
        self._graph_block_table: Optional[torch.Tensor] = None
        self._graph_cache_seqlens: Optional[torch.Tensor] = None

    # ======================================================================
    # Device resolution
    # ======================================================================

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available — falling back to cpu")
                return "cpu"
            if ":" in device:
                ordinal = int(device.split(":")[1])
                if ordinal >= torch.cuda.device_count():
                    logger.warning(
                        "Device %r not available (%d GPU(s)) — falling back to cpu",
                        device, torch.cuda.device_count(),
                    )
                    return "cpu"
            return device
        try:
            torch.empty(1, device=device)
            return device
        except (RuntimeError, AssertionError):
            logger.warning("Device %r unavailable — falling back to cpu", device)
            return "cpu"

    # ======================================================================
    # Properties
    # ======================================================================

    @property
    def num_free_blocks(self) -> int:
        return self.pool.get_num_free_blocks()

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.pool.get_num_free_blocks()

    # ======================================================================
    # LRU tracking (sequence level)
    # ======================================================================

    def _touch(self, seq_id: int) -> None:
        """Update LRU access tick. O(log n) heap push."""
        self._access_counter += 1
        self._last_access[seq_id] = self._access_counter
        heapq.heappush(self._lru_heap, (self._access_counter, seq_id))
        active_count = len(self._seq_blocks)
        if len(self._lru_heap) > max(256, 4 * active_count):
            self._compact_lru_heap()

    def _compact_lru_heap(self) -> None:
        fresh = [
            (tick, sid) for tick, sid in self._lru_heap
            if sid in self._seq_blocks
            and self._last_access.get(sid) == tick
        ]
        heapq.heapify(fresh)
        self._lru_heap = fresh

    def _evict_lru(self, num_blocks_needed: int, protect_seq_id: int = -1) -> int:
        """
        Evict LRU sequences until num_blocks_needed free blocks are available.
        Returns total blocks freed.
        """
        freed = 0
        self._evicted_seq_ids.clear()

        while freed < num_blocks_needed and self._lru_heap:
            tick, victim = heapq.heappop(self._lru_heap)

            if victim not in self._seq_blocks:
                continue
            if self._last_access.get(victim, 0) != tick:
                continue
            if victim == protect_seq_id or victim in self._pinned_seq_ids:
                heapq.heappush(self._lru_heap, (tick, victim))
                continue

            before = self.pool.get_num_free_blocks()
            self.free_sequence(victim)
            freed += self.pool.get_num_free_blocks() - before
            self._eviction_count += 1
            self._evicted_seq_ids.append(victim)

        return freed

    def get_evicted_seq_ids(self) -> List[int]:
        return list(self._evicted_seq_ids)

    # ======================================================================
    # Block allocation
    # ======================================================================

    def allocate_blocks(self, seq_id: int, num_blocks_needed: int) -> List[int]:
        """
        Allocate blocks for a sequence.

        Uses BlockPool.get_new_blocks() which evicts from the LRU tail of the
        free queue.  If the pool is still short (prefix-cached blocks are being
        held by other seqs), falls back to sequence-level LRU eviction.

        Returns list of newly allocated physical block IDs.
        """
        if num_blocks_needed > self.pool.get_num_free_blocks():
            if self.enable_lru_eviction:
                deficit = num_blocks_needed - self.pool.get_num_free_blocks()
                freed = self._evict_lru(deficit, protect_seq_id=seq_id)
                if freed < deficit:
                    raise RuntimeError(
                        f"OOM after LRU eviction: need {num_blocks_needed} blocks, "
                        f"freed {freed}, have {self.pool.get_num_free_blocks()}"
                    )
            else:
                raise RuntimeError(
                    f"OOM: need {num_blocks_needed} blocks, "
                    f"have {self.pool.get_num_free_blocks()}"
                )

        new_blocks = self.pool.get_new_blocks(num_blocks_needed)

        # Lazy zeroing: zero GPU memory for blocks that were previously used
        for blk in new_blocks:
            if blk.block_id in self._dirty_blocks:
                self._zero_block(blk.block_id)
                self._dirty_blocks.discard(blk.block_id)

        existing = self._seq_blocks.get(seq_id, [])
        start = len(existing)
        for i, blk in enumerate(new_blocks):
            self.block_table[seq_id, start + i] = blk.block_id
        self._seq_blocks[seq_id] = existing + new_blocks

        self._touch(seq_id)
        return [blk.block_id for blk in new_blocks]

    # ======================================================================
    # KV read / write
    # ======================================================================

    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        position: int,
        k: torch.Tensor,   # (num_kv_heads, head_dim)
        v: torch.Tensor,   # (num_kv_heads, head_dim)
    ) -> None:
        block_idx = position // self.block_size
        offset    = position % self.block_size
        physical  = self.block_table[seq_id, block_idx].item()

        if physical < 0:
            [physical] = self.allocate_blocks(seq_id, 1)

        physical = self._cow_if_shared(seq_id, block_idx, physical)

        k_kv = k if k.dtype == self.kv_dtype else k.to(self.kv_dtype)
        v_kv = v if v.dtype == self.kv_dtype else v.to(self.kv_dtype)
        self.k_caches[layer_idx][physical, offset] = k_kv
        self.v_caches[layer_idx][physical, offset] = v_kv
        self.seq_lens[seq_id] = max(self.seq_lens[seq_id].item(), position + 1)
        self._touch(seq_id)

    def read_kv(
        self,
        layer_idx: int,
        seq_id: int,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read full K/V for a sequence. Returns (seq_len, num_kv_heads, head_dim)."""
        self._touch(seq_id)
        seq_len = self.seq_lens[seq_id].item()
        if max_len is not None:
            seq_len = min(seq_len, max_len)

        empty = lambda: torch.zeros(
            0, self.num_kv_heads, self.head_dim,
            dtype=self.compute_dtype, device=self.device,
        )
        if seq_len == 0:
            return empty(), empty()

        k_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim,
                            dtype=self.compute_dtype, device=self.device)
        v_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim,
                            dtype=self.compute_dtype, device=self.device)

        num_full = seq_len // self.block_size
        remainder = seq_len % self.block_size
        total = num_full + (1 if remainder else 0)
        physical_ids = self.block_table[seq_id, :total].tolist() if total else []

        need_cast = self.kv_dtype != self.compute_dtype
        for bidx in range(num_full):
            pb = physical_ids[bidx]
            if pb >= 0:
                s, e = bidx * self.block_size, (bidx + 1) * self.block_size
                k_b = self.k_caches[layer_idx][pb, :self.block_size]
                v_b = self.v_caches[layer_idx][pb, :self.block_size]
                k_out[s:e] = k_b.to(self.compute_dtype) if need_cast else k_b
                v_out[s:e] = v_b.to(self.compute_dtype) if need_cast else v_b

        if remainder:
            pb = physical_ids[num_full]
            if pb >= 0:
                s = num_full * self.block_size
                k_b = self.k_caches[layer_idx][pb, :remainder]
                v_b = self.v_caches[layer_idx][pb, :remainder]
                k_out[s:seq_len] = k_b.to(self.compute_dtype) if need_cast else k_b
                v_out[s:seq_len] = v_b.to(self.compute_dtype) if need_cast else v_b

        return k_out, v_out

    def write_kv_batch(
        self,
        layer_idx: int,
        seq_id: int,
        positions: torch.Tensor,   # (n,) int32
        k: torch.Tensor,           # (n, num_kv_heads, head_dim)
        v: torch.Tensor,           # (n, num_kv_heads, head_dim)
    ) -> None:
        """Batch write for one sequence (vectorised block-table lookups)."""
        n = positions.shape[0]
        if n == 0:
            return

        pos_np       = positions.cpu().numpy()
        block_indices = pos_np // self.block_size
        offsets      = pos_np % self.block_size

        unique_bidx = sorted(set(int(b) for b in block_indices))
        block_vals  = self.block_table[seq_id, unique_bidx].tolist()
        missing     = sum(1 for bv in block_vals if bv < 0)
        if missing:
            self.allocate_blocks(seq_id, missing)
            block_vals = self.block_table[seq_id, unique_bidx].tolist()

        for i, bidx in enumerate(unique_bidx):
            if block_vals[i] >= 0:
                self._cow_if_shared(seq_id, bidx, block_vals[i])
        block_vals = self.block_table[seq_id, unique_bidx].tolist()
        block_map  = {bidx: block_vals[i] for i, bidx in enumerate(unique_bidx)}

        k_kv = k if k.dtype == self.kv_dtype else k.to(self.kv_dtype)
        v_kv = v if v.dtype == self.kv_dtype else v.to(self.kv_dtype)

        if n <= 4:
            for t in range(n):
                pb  = block_map[int(block_indices[t])]
                off = int(offsets[t])
                self.k_caches[layer_idx][pb, off] = k_kv[t]
                self.v_caches[layer_idx][pb, off] = v_kv[t]
        else:
            from collections import defaultdict
            groups: dict = defaultdict(list)
            for t in range(n):
                groups[block_map[int(block_indices[t])]].append((int(offsets[t]), t))
            for pb, entries in groups.items():
                offs = [e[0] for e in entries]
                idxs = [e[1] for e in entries]
                self.k_caches[layer_idx][pb, offs] = k_kv[idxs]
                self.v_caches[layer_idx][pb, offs] = v_kv[idxs]

        max_pos = int(pos_np.max()) + 1
        self.seq_lens[seq_id] = max(int(self.seq_lens[seq_id]), max_pos)
        self._touch(seq_id)

    def write_kv_decode(
        self,
        layer_idx: int,
        seq_ids_tensor: torch.Tensor,  # (batch,) long
        positions: torch.Tensor,        # (batch,) long
        k: torch.Tensor,                # (batch, num_kv_heads, head_dim)
        v: torch.Tensor,                # (batch, num_kv_heads, head_dim)
    ) -> None:
        """
        Tensor-only KV write for decode (1 token per seq). CUDA-graph compatible.
        No Python loops, no .item() calls.  Blocks must already be allocated.
        """
        block_idx = (positions // self.block_size).long()
        offset    = (positions % self.block_size).long()

        if self._graph_mode and self._graph_block_table is not None:
            n        = seq_ids_tensor.shape[0]
            physical = self._graph_block_table[self._graph_batch_range[:n], block_idx]
        else:
            physical = self.block_table[seq_ids_tensor, block_idx]

        physical = physical.clamp(min=0)

        k_kv = k if k.dtype == self.kv_dtype else k.to(self.kv_dtype)
        v_kv = v if v.dtype == self.kv_dtype else v.to(self.kv_dtype)
        self.k_caches[layer_idx][physical, offset] = k_kv
        self.v_caches[layer_idx][physical, offset] = v_kv

        new_lens = (positions + 1).to(torch.int32)
        if self._graph_mode and self._graph_cache_seqlens is not None:
            n = seq_ids_tensor.shape[0]
            self._graph_cache_seqlens[:n] = torch.maximum(
                self._graph_cache_seqlens[:n], new_lens
            )
        else:
            self.seq_lens[seq_ids_tensor] = torch.maximum(
                self.seq_lens[seq_ids_tensor], new_lens
            )

    # ======================================================================
    # Cache tensor accessors (for attention kernels)
    # ======================================================================

    def get_cache_tensors(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw (k_cache, v_cache) for a layer — flash_attn compatible."""
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def get_block_table_for_seqs(self, seq_ids: List[int]) -> torch.Tensor:
        if self._graph_mode and self._graph_block_table is not None:
            return self._graph_block_table[:len(seq_ids)]
        return self.block_table[seq_ids]

    def get_cache_seqlens(self, seq_ids: List[int]) -> torch.Tensor:
        if self._graph_mode and self._graph_cache_seqlens is not None:
            return self._graph_cache_seqlens[:len(seq_ids)]
        return self.seq_lens[seq_ids]

    def get_block_table_for_seqs_tensor(
        self, seq_ids_tensor: torch.Tensor
    ) -> torch.Tensor:
        if self._graph_mode and self._graph_block_table is not None:
            return self._graph_block_table[:seq_ids_tensor.shape[0]]
        return self.block_table[seq_ids_tensor]

    def get_cache_seqlens_tensor(
        self, seq_ids_tensor: torch.Tensor
    ) -> torch.Tensor:
        if self._graph_mode and self._graph_cache_seqlens is not None:
            return self._graph_cache_seqlens[:seq_ids_tensor.shape[0]]
        return self.seq_lens[seq_ids_tensor]

    # ======================================================================
    # Sequence lifecycle
    # ======================================================================

    def free_sequence(self, seq_id: int) -> None:
        """
        Free all blocks for a sequence.

        Blocks are freed in reverse order (tail first) so the most-recently-
        written blocks stay at the tail of the free queue — maximising prefix
        cache reuse for future requests with the same prefix.
        """
        blocks = self._seq_blocks.pop(seq_id, [])
        if not blocks:
            self.seq_lens[seq_id] = 0
            self._last_access.pop(seq_id, None)
            return

        # Mark GPU blocks dirty (lazy zeroing at next allocation)
        for blk in blocks:
            self._dirty_blocks.add(blk.block_id)

        # Free via pool (reverse = tail first, better for prefix cache LRU)
        self.pool.free_blocks(reversed(blocks))

        # Clear block table (blocks were stored contiguously from index 0)
        n = len(blocks)
        self.block_table[seq_id, :n] = -1

        self.seq_lens[seq_id] = 0
        self._last_access.pop(seq_id, None)

    # ======================================================================
    # CUDA graph support
    # ======================================================================

    def init_graph_buffers(self, max_batch_size: int) -> None:
        """Allocate static buffers (same tensor addresses across captures)."""
        self._graph_block_table = torch.full(
            (max_batch_size, self.max_blocks_per_seq), 0,
            dtype=torch.int32, device=self.device,
        )
        self._graph_cache_seqlens = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device,
        )
        self._graph_batch_range = torch.arange(
            max_batch_size, dtype=torch.long, device=self.device,
        )

    def enter_graph_mode(self, seq_ids: List[int]) -> None:
        n = len(seq_ids)
        if self._graph_block_table is None:
            return
        self._graph_block_table[:n].copy_(self.block_table[seq_ids])
        self._graph_cache_seqlens[:n].copy_(self.seq_lens[seq_ids])
        self._graph_mode = True

    def exit_graph_mode(self, seq_ids: Optional[List[int]] = None) -> None:
        """
        Disable graph mode and sync updated seqlens back to ground truth.
        (Bug 9 fix: write_kv_decode updates _graph_cache_seqlens precisely.)
        """
        self._graph_mode = False
        if seq_ids is not None and self._graph_cache_seqlens is not None:
            n = len(seq_ids)
            self.seq_lens[seq_ids] = self._graph_cache_seqlens[:n].to(
                self.seq_lens.dtype
            )

    # ======================================================================
    # Prefix caching
    # ======================================================================

    def enable_prefix_caching(self) -> None:
        """Switch the BlockPool to caching mode."""
        self.pool.enable_caching = True

    @property
    def prefix_cache_enabled(self) -> bool:
        return self.pool.enable_caching

    @property
    def _prefix_hash_to_blocks(self) -> dict:
        """Expose BlockPool hash→block mapping for introspection/testing."""
        return self.pool._hash_to_block

    @staticmethod
    def _hash_token_block(token_ids: List[int], prev_hash: Optional[bytes] = None) -> bytes:
        """Deterministic hash for a token block (delegates to BlockPool.hash_block)."""
        return BlockPool.hash_block(token_ids, prev_hash)

    def try_reuse_prefix(self, seq_id: int, token_ids: List[int]) -> int:
        """
        Try to reuse cached prefix blocks.  Returns number of prefix tokens
        reused (always a multiple of block_size).
        """
        if not self.pool.enable_caching:
            return 0

        reused = 0
        prev_hash: Optional[bytes] = None
        num_full_blocks = len(token_ids) // self.block_size

        for bidx in range(num_full_blocks):
            start = bidx * self.block_size
            end   = start + self.block_size
            block_tokens = token_ids[start:end]
            block_hash   = BlockPool.hash_block(block_tokens, prev_hash)

            cached = self.pool.get_cached_block(block_hash)
            if cached is None:
                break

            # Touch the cached block (increments ref_cnt, removes from free queue)
            self.pool.touch([cached])
            self.block_table[seq_id, bidx] = cached.block_id

            existing = self._seq_blocks.get(seq_id, [])
            self._seq_blocks[seq_id] = existing + [cached]

            reused    += self.block_size
            prev_hash  = block_hash

        if reused:
            self.seq_lens[seq_id] = reused
            self._touch(seq_id)

        return reused

    def register_prefix_blocks(self, seq_id: int, token_ids: List[int]) -> None:
        """Register computed KV blocks in the prefix cache for future reuse."""
        if not self.pool.enable_caching:
            return

        blocks = self._seq_blocks.get(seq_id, [])
        num_full = min(len(token_ids) // self.block_size, len(blocks))
        if num_full == 0:
            return

        prev_hash: Optional[bytes] = None
        for bidx in range(num_full):
            start = bidx * self.block_size
            end   = start + self.block_size
            h     = BlockPool.hash_block(token_ids[start:end], prev_hash)
            self.pool.cache_block(blocks[bidx], h)
            prev_hash = h

    def _cow_if_shared(
        self, seq_id: int, block_idx: int, physical_block: int
    ) -> int:
        """Copy-on-write: if block is shared via prefix cache, make a private copy."""
        if not self.pool.enable_caching:
            return physical_block

        blocks = self._seq_blocks.get(seq_id)
        if blocks is None or block_idx >= len(blocks):
            return physical_block

        blk = blocks[block_idx]
        if blk.ref_cnt <= 1:
            return physical_block  # sole owner

        if self.pool.get_num_free_blocks() == 0:
            return physical_block  # can't copy

        [new_blk] = self.pool.get_new_blocks(1)
        for layer in range(self.num_layers):
            self.k_caches[layer][new_blk.block_id].copy_(
                self.k_caches[layer][physical_block]
            )
            self.v_caches[layer][new_blk.block_id].copy_(
                self.v_caches[layer][physical_block]
            )
        self.block_table[seq_id, block_idx] = new_blk.block_id
        # Decrement ref on the shared block and add new private block to seq
        blk.ref_cnt -= 1
        if blk.ref_cnt == 0 and not blk.is_null:
            self.pool.free_block_queue.append(blk)
        blocks[block_idx] = new_blk
        return new_blk.block_id

    # ======================================================================
    # Lazy block zeroing
    # ======================================================================

    def _zero_block(self, block_id: int) -> None:
        """Zero out GPU KV memory for a block (called lazily at next alloc)."""
        for layer in range(self.num_layers):
            self.k_caches[layer][block_id].zero_()
            self.v_caches[layer][block_id].zero_()

    # ======================================================================
    # Swap-to-CPU
    # ======================================================================

    def enable_swap(self) -> None:
        """Allocate CPU pinned memory mirror."""
        if self.device == "cpu":
            return
        self._swap_enabled = True
        can_pin = torch.cuda.is_available()

        def _cpu_cache():
            t = torch.zeros(
                self.num_blocks, self.block_size,
                self.num_kv_heads, self.head_dim,
                dtype=self.kv_dtype, device="cpu",
            )
            return t.pin_memory() if can_pin else t

        self._cpu_k_caches = [_cpu_cache() for _ in range(self.num_layers)]
        self._cpu_v_caches = [_cpu_cache() for _ in range(self.num_layers)]
        self._cpu_free_blocks = list(range(self.num_blocks))
        self._cpu_block_table = torch.full(
            (self.max_seqs, self.max_blocks_per_seq), -1,
            dtype=torch.int32, device="cpu",
        )

    @property
    def swap_enabled(self) -> bool:
        return self._swap_enabled

    def swap_out(self, seq_id: int) -> bool:
        if not self._swap_enabled:
            return False
        blocks = self._seq_blocks.get(seq_id, [])
        if not blocks or len(self._cpu_free_blocks) < len(blocks):
            return False

        seq_len    = int(self.seq_lens[seq_id])
        cpu_blocks = []
        stream     = torch.cuda.Stream() if torch.cuda.is_available() else None

        ctx = torch.cuda.stream(stream) if stream else _nullcontext()
        with ctx:
            for i, blk in enumerate(blocks):
                cpu_id = self._cpu_free_blocks.pop()
                cpu_blocks.append(cpu_id)
                for layer in range(self.num_layers):
                    self._cpu_k_caches[layer][cpu_id].copy_(
                        self.k_caches[layer][blk.block_id], non_blocking=True
                    )
                    self._cpu_v_caches[layer][cpu_id].copy_(
                        self.v_caches[layer][blk.block_id], non_blocking=True
                    )
                self._cpu_block_table[seq_id, i] = cpu_id
                self._dirty_blocks.add(blk.block_id)
                self.block_table[seq_id, i] = -1

        if stream:
            stream.synchronize()

        self.pool.free_blocks(reversed(blocks))
        self._swapped_seqs[seq_id] = {
            "cpu_blocks": cpu_blocks,
            "seq_len": seq_len,
            "num_blocks": len(blocks),
        }
        del self._seq_blocks[seq_id]
        self.seq_lens[seq_id] = 0
        self._swap_count += 1
        return True

    def swap_in(self, seq_id: int) -> bool:
        if seq_id not in self._swapped_seqs:
            return False
        meta      = self._swapped_seqs[seq_id]
        cpu_ids   = meta["cpu_blocks"]
        n_blocks  = len(cpu_ids)

        if self.pool.get_num_free_blocks() < n_blocks:
            return False

        new_blocks = self.pool.get_new_blocks(n_blocks)
        stream     = torch.cuda.Stream() if torch.cuda.is_available() else None

        ctx = torch.cuda.stream(stream) if stream else _nullcontext()
        with ctx:
            for i, (cpu_id, blk) in enumerate(zip(cpu_ids, new_blocks)):
                for layer in range(self.num_layers):
                    self.k_caches[layer][blk.block_id].copy_(
                        self._cpu_k_caches[layer][cpu_id], non_blocking=True
                    )
                    self.v_caches[layer][blk.block_id].copy_(
                        self._cpu_v_caches[layer][cpu_id], non_blocking=True
                    )
                self.block_table[seq_id, i] = blk.block_id
                self._cpu_block_table[seq_id, i] = -1
                self._cpu_free_blocks.append(cpu_id)

        if stream:
            stream.synchronize()

        self._seq_blocks[seq_id] = new_blocks
        self.seq_lens[seq_id]    = meta["seq_len"]
        del self._swapped_seqs[seq_id]
        self._touch(seq_id)
        return True

    # ======================================================================
    # FP8 auto-upgrade
    # ======================================================================

    def maybe_enable_fp8(self, utilization_threshold: float = 0.70) -> bool:
        if self.device == "cpu":
            return False
        if self.kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return True
        if self.pool.get_usage() < utilization_threshold:
            return False
        target = torch.float8_e4m3fn
        for layer in range(self.num_layers):
            self.k_caches[layer] = self.k_caches[layer].to(target)
            self.v_caches[layer] = self.v_caches[layer].to(target)
        self.kv_dtype = target
        return True

    # ======================================================================
    # Stats
    # ======================================================================

    def get_stats(self) -> dict:
        stats = {
            "num_blocks":  self.num_blocks,
            "used_blocks": self.num_used_blocks,
            "free_blocks": self.num_free_blocks,
            "block_size":  self.block_size,
            "active_seqs": len(self._seq_blocks),
            "pool_usage":  round(self.pool.get_usage(), 3),
        }
        if self.pool.enable_caching:
            stats["prefix_cached_blocks"] = len(self.pool._hash_to_block)
            stats["prefix_unique_hashes"] = len(self.pool._hash_to_block)
        if self.enable_lru_eviction:
            stats["lru_evictions_total"] = self._eviction_count
        if self._swap_enabled:
            stats["swapped_seqs"]     = len(self._swapped_seqs)
            stats["cpu_free_blocks"]  = len(self._cpu_free_blocks)
            stats["swap_count_total"] = self._swap_count
        return stats


# ---------------------------------------------------------------------------
# Minimal context manager for when no CUDA stream is needed
# ---------------------------------------------------------------------------

class _nullcontext:
    def __enter__(self):  return self
    def __exit__(self, *a): pass
