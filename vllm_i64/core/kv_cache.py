"""
vllm-i64 :: Paged KV Cache

Block-based KV cache with integer indexing.

All metadata is integer:
  - Block table: i32 (block_id per slot)
  - Free list: i32 (available block IDs)
  - Sequence lengths: i32

Only the K/V tensors themselves are float (they store attention states).

INL - 2025
"""

import torch
import heapq
import hashlib
import logging
from typing import Optional, Tuple, List, Dict, Set

logger = logging.getLogger(__name__)


class PagedKVCache:
    """
    Paged KV cache with integer block management.

    Memory layout (flash_attn-compatible):
        k_cache: (num_blocks, block_size, num_heads, head_dim) float
        v_cache: (num_blocks, block_size, num_heads, head_dim) float
        block_table: (max_seqs, max_blocks_per_seq) i32

    Block allocation/deallocation is pure integer.

    LRU eviction: when the cache is full, the least recently used
    sequence is evicted to make room — no RuntimeError, no lost requests.
    All LRU tracking is integer (monotonic counter).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        max_seqs: int = 64,
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

        # Device validation with CPU fallback
        self.device = self._resolve_device(device)

        # FP8 quantization: store KV in FP8 for 2x memory savings
        if kv_cache_dtype == "fp8" and self.device != "cpu":
            self.kv_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2" and self.device != "cpu":
            self.kv_dtype = torch.float8_e5m2
        else:
            self.kv_dtype = dtype
        self.dtype = dtype  # backward compat (read_kv returns this dtype)

        # Max blocks any single sequence can hold — any seq can use all blocks;
        # LRU eviction handles fair sharing across sequences.
        self.max_blocks_per_seq = num_blocks

        # KV tensors (stored in kv_dtype — may be FP8 for memory savings)
        # Per layer: (num_blocks, block_size, num_kv_heads, head_dim)
        # Layout matches flash_attn_with_kv_cache expectations
        self.k_caches = [
            torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=self.kv_dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=self.kv_dtype, device=device)
            for _ in range(num_layers)
        ]

        # Block table: maps (seq_id, block_idx) → physical block_id (i32)
        # On device (GPU when available) to avoid CPU→GPU transfer on decode hot path.
        # Management code uses .item() for scalar reads (sync is fine, not on hot path).
        self.block_table = torch.full(
            (max_seqs, self.max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )

        # Free block list (integer)
        self.free_blocks: List[int] = list(range(num_blocks))

        # Sequence lengths (integer) — on device for flash_attn cache_seqlens
        self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32, device=self.device)

        # ── LRU eviction tracking (all integer, heap-based O(log n)) ──
        self.enable_lru_eviction = enable_lru_eviction
        self._access_counter: int = 0          # monotonic i64 clock
        self._last_access: Dict[int, int] = {} # seq_id → last access tick
        self._eviction_count: int = 0          # total evictions performed
        self._evicted_seq_ids: List[int] = []  # recently evicted (for scheduler)
        self._active_seq_ids: set = set()      # O(1) tracking of allocated seqs
        self._pinned_seq_ids: set = set()      # Sequences protected from eviction
        # Min-heap of (tick, seq_id) for O(log n) eviction candidate selection
        self._lru_heap: List[Tuple[int, int]] = []

        # ── Swap-to-CPU ──
        self._swap_enabled = False
        self._cpu_k_caches: Optional[List[torch.Tensor]] = None
        self._cpu_v_caches: Optional[List[torch.Tensor]] = None
        self._cpu_free_blocks: List[int] = []
        self._cpu_block_table: Optional[torch.Tensor] = None
        self._swapped_seqs: Dict[int, dict] = {}  # seq_id → swap metadata
        self._swap_count: int = 0

        # ── O(1) block count per sequence ──
        # Avoids scanning block_table[seq_id] >= 0 on every allocate/evict.
        self._block_count: Dict[int, int] = {}

        # ── Lazy block zeroing ──
        # Blocks are zeroed on allocation, not on free. This makes
        # free_sequence() O(1) per block and defers the GPU write to
        # the next allocation — beneficial for bulk LRU evictions.
        self._dirty_blocks: Set[int] = set()

        # ── CUDA graph support ──
        # Static buffers for block_table/seqlens used during graph replay.
        # These keep the same tensor addresses across captures and replays.
        self._graph_mode: bool = False
        self._graph_block_table: Optional[torch.Tensor] = None
        self._graph_cache_seqlens: Optional[torch.Tensor] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Validate device and fall back to CPU if unavailable."""
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(
                    "Device %r requested but CUDA is not available — falling back to cpu",
                    device,
                )
                return "cpu"
            # Validate specific device ordinal (e.g. "cuda:3")
            if ":" in device:
                ordinal = int(device.split(":")[1])
                if ordinal >= torch.cuda.device_count():
                    logger.warning(
                        "Device %r requested but only %d GPU(s) visible — falling back to cpu",
                        device, torch.cuda.device_count(),
                    )
                    return "cpu"
            return device
        # Unknown device string — let PyTorch try, fall back on failure
        try:
            torch.empty(1, device=device)
            return device
        except (RuntimeError, AssertionError):
            logger.warning(
                "Device %r is not available — falling back to cpu", device,
            )
            return "cpu"

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def _touch(self, seq_id: int):
        """Update LRU access tick for a sequence. Pure integer + heap push O(log n)."""
        self._access_counter += 1
        self._last_access[seq_id] = self._access_counter
        heapq.heappush(self._lru_heap, (self._access_counter, seq_id))
        # Periodic compaction: if heap has >4x active entries, rebuild
        if len(self._lru_heap) > max(256, 4 * len(self._active_seq_ids)):
            self._compact_lru_heap()

    def _compact_lru_heap(self):
        """Remove stale entries from the LRU heap."""
        fresh = [
            (tick, sid) for tick, sid in self._lru_heap
            if sid in self._active_seq_ids and self._last_access.get(sid) == tick
        ]
        heapq.heapify(fresh)
        self._lru_heap = fresh

    def _evict_lru(self, num_blocks_needed: int, protect_seq_id: int = -1) -> int:
        """
        Evict least recently used sequences until we have enough free blocks.
        Uses a min-heap for O(log n) candidate selection instead of O(n log n) sort.

        Args:
            num_blocks_needed: how many blocks we need to free
            protect_seq_id: don't evict this sequence (the one requesting blocks)

        Returns:
            Number of blocks freed.
        """
        freed = 0
        self._evicted_seq_ids.clear()

        # Pop from min-heap to find LRU candidates in O(log n) per eviction
        while freed < num_blocks_needed and self._lru_heap:
            tick, victim_sid = heapq.heappop(self._lru_heap)

            # Skip stale entries: seq no longer active, or tick is outdated
            if victim_sid not in self._active_seq_ids:
                continue
            if self._last_access.get(victim_sid, 0) != tick:
                continue  # Stale — this seq was touched again after this entry
            if victim_sid == protect_seq_id or victim_sid in self._pinned_seq_ids:
                # Re-push protected/pinned seq so it's not lost from the heap
                heapq.heappush(self._lru_heap, (tick, victim_sid))
                continue

            # Measure actual freed blocks via free_blocks diff
            free_before = len(self.free_blocks)
            self.free_sequence(victim_sid)
            freed += len(self.free_blocks) - free_before
            self._eviction_count += 1
            self._evicted_seq_ids.append(victim_sid)

        return freed

    def get_evicted_seq_ids(self) -> List[int]:
        """Return seq_ids evicted in the last allocate_blocks call (for scheduler sync)."""
        return list(self._evicted_seq_ids)

    def allocate_blocks(self, seq_id: int, num_blocks_needed: int) -> List[int]:
        """
        Allocate blocks for a sequence. Pure integer operation.

        With LRU eviction enabled, automatically evicts the least recently
        used sequences when the cache is full instead of raising OOM.

        Returns list of allocated block IDs.
        """
        if num_blocks_needed > len(self.free_blocks):
            if self.enable_lru_eviction:
                deficit = num_blocks_needed - len(self.free_blocks)
                freed = self._evict_lru(deficit, protect_seq_id=seq_id)
                if freed < deficit:
                    raise RuntimeError(
                        f"OOM after LRU eviction: need {num_blocks_needed} blocks, "
                        f"freed {freed}, have {len(self.free_blocks)}"
                    )
            else:
                raise RuntimeError(
                    f"OOM: need {num_blocks_needed} blocks, have {len(self.free_blocks)}"
                )

        allocated = []
        current_blocks = self._block_count.get(seq_id, 0)

        for i in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            # Lazy zeroing: only zero blocks that were previously used
            if block_id in self._dirty_blocks:
                self._zero_block(block_id)
                self._dirty_blocks.discard(block_id)
            block_idx = current_blocks + i
            self.block_table[seq_id, block_idx] = block_id
            allocated.append(block_id)

        self._block_count[seq_id] = current_blocks + num_blocks_needed
        self._touch(seq_id)
        self._active_seq_ids.add(seq_id)
        return allocated


    def write_kv(
        self,
        layer_idx: int,
        seq_id: int,
        position: int,
        k: torch.Tensor,   # (num_kv_heads, head_dim)
        v: torch.Tensor,   # (num_kv_heads, head_dim)
    ):
        """
        Write K/V to cache at a position.

        Block lookup is integer:
            block_idx = position // block_size
            offset = position % block_size
            physical_block = block_table[seq_id, block_idx]
        """
        block_idx = position // self.block_size
        offset = position % self.block_size
        physical_block = self.block_table[seq_id, block_idx].item()

        if physical_block < 0:
            # Need to allocate
            [physical_block] = self.allocate_blocks(seq_id, 1)

        # Copy-on-write: if this block is shared via prefix caching, copy first
        physical_block = self._cow_if_shared(seq_id, block_idx, physical_block)

        k_kv = k if k.dtype == self.kv_dtype else k.to(self.kv_dtype)
        v_kv = v if v.dtype == self.kv_dtype else v.to(self.kv_dtype)
        self.k_caches[layer_idx][physical_block, offset, :, :] = k_kv
        self.v_caches[layer_idx][physical_block, offset, :, :] = v_kv
        self.seq_lens[seq_id] = max(self.seq_lens[seq_id].item(), position + 1)
        self._touch(seq_id)

    def read_kv(
        self,
        layer_idx: int,
        seq_id: int,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read all K/V for a sequence. Vectorized by block to avoid per-token Python loops.

        Returns:
            k: (seq_len, num_kv_heads, head_dim)
            v: (seq_len, num_kv_heads, head_dim)
        """
        self._touch(seq_id)
        seq_len = self.seq_lens[seq_id].item()
        if max_len is not None:
            seq_len = min(seq_len, max_len)

        if seq_len == 0:
            return (
                torch.zeros(0, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.device),
                torch.zeros(0, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.device),
            )

        # Always return in compute dtype (dequantize if FP8)
        k_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.device)
        v_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.device)

        # Vectorized by block — copy entire block slices instead of per-token
        num_full_blocks = seq_len // self.block_size
        remainder = seq_len % self.block_size
        total_blocks = num_full_blocks + (1 if remainder > 0 else 0)

        # Fetch all physical block IDs in one tensor op (1 GPU→CPU transfer)
        if total_blocks > 0:
            physical_blocks = self.block_table[seq_id, :total_blocks].tolist()
        else:
            physical_blocks = []

        need_cast = self.kv_dtype != self.compute_dtype
        for block_idx in range(num_full_blocks):
            physical_block = physical_blocks[block_idx]
            if physical_block >= 0:
                start = block_idx * self.block_size
                end = start + self.block_size
                k_block = self.k_caches[layer_idx][physical_block, :self.block_size, :, :]
                v_block = self.v_caches[layer_idx][physical_block, :self.block_size, :, :]
                k_out[start:end] = k_block.to(self.compute_dtype) if need_cast else k_block
                v_out[start:end] = v_block.to(self.compute_dtype) if need_cast else v_block

        if remainder > 0:
            physical_block = physical_blocks[num_full_blocks]
            if physical_block >= 0:
                start = num_full_blocks * self.block_size
                k_block = self.k_caches[layer_idx][physical_block, :remainder, :, :]
                v_block = self.v_caches[layer_idx][physical_block, :remainder, :, :]
                k_out[start:seq_len] = k_block.to(self.compute_dtype) if need_cast else k_block
                v_out[start:seq_len] = v_block.to(self.compute_dtype) if need_cast else v_block

        return k_out, v_out

    def write_kv_batch(
        self,
        layer_idx: int,
        seq_id: int,
        positions: torch.Tensor,   # (n,) int32
        k: torch.Tensor,           # (n, num_kv_heads, head_dim)
        v: torch.Tensor,           # (n, num_kv_heads, head_dim)
    ):
        """Batch write K/V for one sequence. Vectorized block table lookups."""
        n = positions.shape[0]
        if n == 0:
            return

        pos_np = positions.cpu().numpy()
        block_indices = pos_np // self.block_size
        offsets = pos_np % self.block_size

        # Pre-allocate any missing blocks — single batch read (1 GPU→CPU transfer)
        unique_blocks = sorted(set(int(b) for b in block_indices))
        block_vals = self.block_table[seq_id, unique_blocks].tolist()
        missing = sum(1 for v in block_vals if v < 0)
        if missing > 0:
            self.allocate_blocks(seq_id, missing)
            # Re-read after allocation
            block_vals = self.block_table[seq_id, unique_blocks].tolist()

        # Copy-on-write: ensure shared prefix blocks get private copies
        for i, bidx in enumerate(unique_blocks):
            if block_vals[i] >= 0:
                self._cow_if_shared(seq_id, bidx, block_vals[i])
        # Re-read after potential CoW copies
        block_vals = self.block_table[seq_id, unique_blocks].tolist()
        block_map = {bidx: block_vals[i] for i, bidx in enumerate(unique_blocks)}

        # Write using cached block map — skip dtype conversion if already correct
        k_kv = k if k.dtype == self.kv_dtype else k.to(self.kv_dtype)
        v_kv = v if v.dtype == self.kv_dtype else v.to(self.kv_dtype)
        if n <= 4:
            # Small batch: direct writes are faster than grouping overhead
            for t in range(n):
                physical_block = block_map[int(block_indices[t])]
                off = int(offsets[t])
                self.k_caches[layer_idx][physical_block, off, :, :] = k_kv[t]
                self.v_caches[layer_idx][physical_block, off, :, :] = v_kv[t]
        else:
            # Batch by physical block: group writes to same block
            from collections import defaultdict
            block_groups: dict = defaultdict(list)
            for t in range(n):
                pb = block_map[int(block_indices[t])]
                block_groups[pb].append((int(offsets[t]), t))
            for pb, entries in block_groups.items():
                offs = [e[0] for e in entries]
                idxs = [e[1] for e in entries]
                self.k_caches[layer_idx][pb, offs, :, :] = k_kv[idxs]
                self.v_caches[layer_idx][pb, offs, :, :] = v_kv[idxs]

        max_pos = int(pos_np.max()) + 1
        cur_len = int(self.seq_lens[seq_id])
        self.seq_lens[seq_id] = max(cur_len, max_pos)
        self._touch(seq_id)

    def get_cache_tensors(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw k/v cache tensors for a layer.
        Shape: (num_blocks, block_size, num_kv_heads, head_dim) — flash_attn compatible.
        """
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def get_block_table_for_seqs(self, seq_ids: List[int]) -> torch.Tensor:
        """Extract block table rows for given sequences. Returns (num_seqs, max_blocks_per_seq) int32."""
        if self._graph_mode and self._graph_block_table is not None:
            return self._graph_block_table[:len(seq_ids)]
        return self.block_table[seq_ids]

    def get_cache_seqlens(self, seq_ids: List[int]) -> torch.Tensor:
        """Get current sequence lengths. Returns (num_seqs,) int32."""
        if self._graph_mode and self._graph_cache_seqlens is not None:
            return self._graph_cache_seqlens[:len(seq_ids)]
        return self.seq_lens[seq_ids]

    def write_kv_decode(
        self,
        layer_idx: int,
        seq_ids_tensor: torch.Tensor,  # (batch,) long — slot indices
        positions: torch.Tensor,        # (batch,) long — write positions
        k: torch.Tensor,                # (batch, num_kv_heads, head_dim)
        v: torch.Tensor,                # (batch, num_kv_heads, head_dim)
    ):
        """
        Tensor-only KV write for decode (1 token per seq). CUDA-graph compatible.

        No Python loops, no .item() calls, no dynamic allocation.
        Blocks must already be allocated (guaranteed for decode after prefill).
        """
        block_idx = (positions // self.block_size).long()
        offset = (positions % self.block_size).long()

        if self._graph_mode and self._graph_block_table is not None:
            n = seq_ids_tensor.shape[0]
            physical = self._graph_block_table[self._graph_batch_range[:n], block_idx]
        else:
            physical = self.block_table[seq_ids_tensor, block_idx]

        physical = physical.clamp(min=0)  # Guard against -1 padding

        k_kv = k if k.dtype == self.kv_dtype else k.to(self.kv_dtype)
        v_kv = v if v.dtype == self.kv_dtype else v.to(self.kv_dtype)
        self.k_caches[layer_idx][physical, offset] = k_kv
        self.v_caches[layer_idx][physical, offset] = v_kv

        # Update seqlens (tensor op, graph-safe)
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

    def get_block_table_for_seqs_tensor(self, seq_ids_tensor: torch.Tensor) -> torch.Tensor:
        """Block table rows for tensor-based seq_ids. Graph-compatible."""
        if self._graph_mode and self._graph_block_table is not None:
            return self._graph_block_table[:seq_ids_tensor.shape[0]]
        return self.block_table[seq_ids_tensor]

    def get_cache_seqlens_tensor(self, seq_ids_tensor: torch.Tensor) -> torch.Tensor:
        """Cache seqlens for tensor-based seq_ids. Graph-compatible."""
        if self._graph_mode and self._graph_cache_seqlens is not None:
            return self._graph_cache_seqlens[:seq_ids_tensor.shape[0]]
        return self.seq_lens[seq_ids_tensor]

    def init_graph_buffers(self, max_batch_size: int):
        """Allocate static buffers for CUDA graph mode (same tensor addresses)."""
        self._graph_block_table = torch.full(
            (max_batch_size, self.max_blocks_per_seq), 0, dtype=torch.int32, device=self.device
        )
        self._graph_cache_seqlens = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device
        )
        self._graph_batch_range = torch.arange(
            max_batch_size, device=self.device, dtype=torch.long
        )

    def enter_graph_mode(self, seq_ids: List[int]):
        """Copy current metadata into static graph buffers and enable graph mode."""
        n = len(seq_ids)
        if self._graph_block_table is None:
            return
        self._graph_block_table[:n].copy_(self.block_table[seq_ids])
        self._graph_cache_seqlens[:n].copy_(self.seq_lens[seq_ids])
        self._graph_mode = True

    def exit_graph_mode(self):
        """Disable graph mode."""
        self._graph_mode = False

    # =====================================================================
    # Prefix Caching — reuse KV blocks across requests with same prefix
    # =====================================================================

    def enable_prefix_caching(self):
        """Enable prefix caching for this cache instance."""
        self._prefix_cache_enabled = True
        # Maps prefix_hash → list of (physical_block_id, layer_idx) that hold this prefix
        self._prefix_hash_to_blocks: Dict[int, List[int]] = {}
        # Maps physical_block_id → prefix_hash (for refcounting)
        self._block_to_prefix: Dict[int, int] = {}
        # Reference count per prefix hash
        self._prefix_refcount: Dict[int, int] = {}
        # Maps prefix_hash → actual token tuple for collision detection
        self._prefix_hash_to_tokens: Dict[int, tuple] = {}

    @property
    def prefix_cache_enabled(self) -> bool:
        return getattr(self, "_prefix_cache_enabled", False)

    @staticmethod
    def _hash_token_block(token_ids: List[int]) -> int:
        """Hash a block of token IDs for prefix matching."""
        h = hashlib.sha256()
        for tid in token_ids:
            h.update(tid.to_bytes(8, "little", signed=True))
        return int.from_bytes(h.digest()[:8], "little")

    def try_reuse_prefix(
        self, seq_id: int, token_ids: List[int]
    ) -> int:
        """
        Try to reuse cached prefix blocks for a new sequence.

        Returns the number of prefix tokens that were reused (always a
        multiple of block_size). The caller can skip computing attention
        for these positions.
        """
        if not self.prefix_cache_enabled:
            return 0

        reused = 0
        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            prefix_hash = self._hash_token_block(block_tokens)

            if prefix_hash in self._prefix_hash_to_blocks:
                # Collision detection: verify actual tokens match
                cached_tokens = self._prefix_hash_to_tokens.get(prefix_hash)
                if cached_tokens is None or cached_tokens != tuple(block_tokens):
                    break  # Hash collision or missing metadata — don't reuse
                # Reuse: point this seq's block table to existing physical block
                physical_block = self._prefix_hash_to_blocks[prefix_hash][0]
                self.block_table[seq_id, block_idx] = physical_block
                self._prefix_refcount[prefix_hash] = self._prefix_refcount.get(prefix_hash, 1) + 1
                reused += self.block_size
            else:
                break  # Can't skip non-contiguous blocks

        if reused > 0:
            self.seq_lens[seq_id] = reused
            # Update block count and LRU tracking so eviction knows about this seq
            num_reused_blocks = reused // self.block_size
            self._block_count[seq_id] = self._block_count.get(seq_id, 0) + num_reused_blocks
            self._active_seq_ids.add(seq_id)
            self._touch(seq_id)

        return reused

    def register_prefix_blocks(self, seq_id: int, token_ids: List[int]):
        """
        After computing a full prefill, register the resulting KV blocks
        as reusable prefix blocks for future requests.
        """
        if not self.prefix_cache_enabled:
            return

        num_full_blocks = len(token_ids) // self.block_size
        if num_full_blocks == 0:
            return

        # Single GPU→CPU transfer for all block IDs
        physical_blocks = self.block_table[seq_id, :num_full_blocks].tolist()

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            prefix_hash = self._hash_token_block(block_tokens)

            physical_block = physical_blocks[block_idx]
            if physical_block >= 0 and prefix_hash not in self._prefix_hash_to_blocks:
                self._prefix_hash_to_blocks[prefix_hash] = [physical_block]
                self._block_to_prefix[physical_block] = prefix_hash
                self._prefix_refcount[prefix_hash] = 1
                self._prefix_hash_to_tokens[prefix_hash] = tuple(block_tokens)

    def _cow_if_shared(self, seq_id: int, block_idx: int, physical_block: int) -> int:
        """Copy-on-write: if block is shared via prefix caching, allocate a private copy."""
        if not self.prefix_cache_enabled:
            return physical_block
        prefix_hash = getattr(self, "_block_to_prefix", {}).get(physical_block)
        if prefix_hash is None:
            return physical_block
        rc = self._prefix_refcount.get(prefix_hash, 1)
        if rc <= 1:
            return physical_block  # Sole owner, no copy needed
        # Allocate a new block and copy KV data
        if not self.free_blocks:
            return physical_block  # Can't copy, no free blocks
        new_block = self.free_blocks.pop()
        if new_block in self._dirty_blocks:
            self._dirty_blocks.discard(new_block)
        for layer_idx in range(self.num_layers):
            self.k_caches[layer_idx][new_block].copy_(self.k_caches[layer_idx][physical_block])
            self.v_caches[layer_idx][new_block].copy_(self.v_caches[layer_idx][physical_block])
        self.block_table[seq_id, block_idx] = new_block
        self._prefix_refcount[prefix_hash] = rc - 1
        return new_block

    def _zero_block(self, block_id: int):
        """Zero out K/V data in a physical block to prevent stale data leaking."""
        for layer_idx in range(self.num_layers):
            self.k_caches[layer_idx][block_id].zero_()
            self.v_caches[layer_idx][block_id].zero_()

    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence. Respects prefix cache refcounts."""
        blocks = self.block_table[seq_id]
        num_blocks = self._block_count.get(seq_id, 0)
        # Fast path: if _block_count says 0, no blocks to free
        if num_blocks == 0:
            self.seq_lens[seq_id] = 0
            self._last_access.pop(seq_id, None)
            self._active_seq_ids.discard(seq_id)
            return
        # Single GPU→CPU transfer for all block IDs
        block_ids = blocks[:num_blocks].tolist()
        block_to_prefix = getattr(self, "_block_to_prefix", {})
        for block_id in block_ids:
            # Check if this block is a shared prefix block
            prefix_hash = block_to_prefix.get(block_id)
            if prefix_hash is not None:
                rc = self._prefix_refcount.get(prefix_hash, 1) - 1
                if rc <= 0:
                    # Last user — free the block (lazy zero on next alloc)
                    self._prefix_hash_to_blocks.pop(prefix_hash, None)
                    block_to_prefix.pop(block_id, None)
                    self._prefix_refcount.pop(prefix_hash, None)
                    self._dirty_blocks.add(block_id)
                    self.free_blocks.append(block_id)
                else:
                    self._prefix_refcount[prefix_hash] = rc
            else:
                self._dirty_blocks.add(block_id)
                self.free_blocks.append(block_id)
        blocks[:num_blocks] = -1

        self._block_count.pop(seq_id, None)
        self.seq_lens[seq_id] = 0
        self._last_access.pop(seq_id, None)
        self._active_seq_ids.discard(seq_id)

    # =====================================================================
    # Swap-to-CPU — preserve KV data in pinned CPU memory
    # =====================================================================

    def enable_swap(self):
        """Allocate CPU pinned memory mirror for swap-to-CPU."""
        if self.device == "cpu":
            return  # No point swapping CPU→CPU
        self._swap_enabled = True
        cpu_num_blocks = self.num_blocks
        can_pin = torch.cuda.is_available()
        self._cpu_k_caches = [
            torch.zeros(
                cpu_num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.kv_dtype, device="cpu",
            ).pin_memory() if can_pin else torch.zeros(
                cpu_num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.kv_dtype, device="cpu",
            )
            for _ in range(self.num_layers)
        ]
        self._cpu_v_caches = [
            torch.zeros(
                cpu_num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.kv_dtype, device="cpu",
            ).pin_memory() if can_pin else torch.zeros(
                cpu_num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                dtype=self.kv_dtype, device="cpu",
            )
            for _ in range(self.num_layers)
        ]
        self._cpu_free_blocks = list(range(cpu_num_blocks))
        self._cpu_block_table = torch.full(
            (self.max_seqs, self.max_blocks_per_seq), -1, dtype=torch.int32, device="cpu"
        )

    @property
    def swap_enabled(self) -> bool:
        return self._swap_enabled

    def swap_out(self, seq_id: int) -> bool:
        """
        Copy KV blocks from GPU to CPU pinned memory, free GPU blocks.
        Uses a dedicated CUDA stream for async D2H copies (pipelined).

        Returns True on success, False if CPU swap space is full.
        """
        if not self._swap_enabled:
            return False

        # O(1) block count lookup
        num_blocks = self._block_count.get(seq_id, 0)
        blocks = self.block_table[seq_id]
        block_list = blocks[:num_blocks].tolist() if num_blocks > 0 else []

        if num_blocks == 0:
            return False
        if len(self._cpu_free_blocks) < num_blocks:
            return False

        seq_len = self.seq_lens[seq_id].item()
        cpu_block_ids = []

        # Use a dedicated CUDA stream for async D2H copy (pipelined)
        swap_stream = None
        if self.device != "cpu" and torch.cuda.is_available():
            swap_stream = torch.cuda.Stream()

        ctx = torch.cuda.stream(swap_stream) if swap_stream else _nullcontext()
        with ctx:
            for i in range(num_blocks):
                gpu_block = block_list[i]
                cpu_block = self._cpu_free_blocks.pop()
                cpu_block_ids.append(cpu_block)

                for layer in range(self.num_layers):
                    self._cpu_k_caches[layer][cpu_block].copy_(self.k_caches[layer][gpu_block], non_blocking=True)
                    self._cpu_v_caches[layer][cpu_block].copy_(self.v_caches[layer][gpu_block], non_blocking=True)

                self._cpu_block_table[seq_id, i] = cpu_block
                self._dirty_blocks.add(gpu_block)
                self.free_blocks.append(gpu_block)
                self.block_table[seq_id, i] = -1

        # Sync the swap stream to ensure copies complete before freeing
        if swap_stream is not None:
            swap_stream.synchronize()

        self._swapped_seqs[seq_id] = {
            "cpu_blocks": cpu_block_ids,
            "seq_len": seq_len,
            "num_blocks": num_blocks,
        }
        self._block_count.pop(seq_id, None)
        self.seq_lens[seq_id] = 0
        self._swap_count += 1
        return True

    def swap_in(self, seq_id: int) -> bool:
        """
        Copy KV blocks from CPU back to GPU.
        Uses a dedicated CUDA stream for async H2D copies (pipelined).

        Returns True on success, False if GPU doesn't have enough free blocks.
        """
        if seq_id not in self._swapped_seqs:
            return False

        meta = self._swapped_seqs[seq_id]
        cpu_block_ids = meta["cpu_blocks"]
        num_blocks = len(cpu_block_ids)

        if len(self.free_blocks) < num_blocks:
            return False

        swap_stream = None
        if self.device != "cpu" and torch.cuda.is_available():
            swap_stream = torch.cuda.Stream()

        ctx = torch.cuda.stream(swap_stream) if swap_stream else _nullcontext()
        with ctx:
            for i, cpu_block in enumerate(cpu_block_ids):
                gpu_block = self.free_blocks.pop()

                for layer in range(self.num_layers):
                    self.k_caches[layer][gpu_block].copy_(self._cpu_k_caches[layer][cpu_block], non_blocking=True)
                    self.v_caches[layer][gpu_block].copy_(self._cpu_v_caches[layer][cpu_block], non_blocking=True)

                self.block_table[seq_id, i] = gpu_block
                self._cpu_block_table[seq_id, i] = -1
                self._cpu_free_blocks.append(cpu_block)

        if swap_stream is not None:
            swap_stream.synchronize()

        self._block_count[seq_id] = num_blocks
        self.seq_lens[seq_id] = meta["seq_len"]
        del self._swapped_seqs[seq_id]
        self._touch(seq_id)
        return True

    def maybe_enable_fp8(self, utilization_threshold: float = 0.70) -> bool:
        """
        Auto-enable FP8 KV cache when memory utilization exceeds threshold.
        Converts existing caches in-place for 2x memory savings.

        Returns True if FP8 was activated (or already active).
        """
        if self.device == "cpu":
            return False
        if self.kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return True  # Already FP8

        utilization = self.num_used_blocks / max(self.num_blocks, 1)
        if utilization < utilization_threshold:
            return False

        # Convert caches to FP8 in-place
        target_dtype = torch.float8_e4m3fn
        for layer_idx in range(self.num_layers):
            self.k_caches[layer_idx] = self.k_caches[layer_idx].to(target_dtype)
            self.v_caches[layer_idx] = self.v_caches[layer_idx].to(target_dtype)
        self.kv_dtype = target_dtype
        return True

    def get_stats(self) -> dict:
        """Cache stats — all integers."""
        stats = {
            "num_blocks": self.num_blocks,
            "used_blocks": self.num_used_blocks,
            "free_blocks": self.num_free_blocks,
            "block_size": self.block_size,
            "active_seqs": int((self.seq_lens > 0).sum().item()),
        }
        if self.prefix_cache_enabled:
            stats["prefix_cached_blocks"] = len(getattr(self, "_block_to_prefix", {}))
            stats["prefix_unique_hashes"] = len(getattr(self, "_prefix_hash_to_blocks", {}))
        if self.enable_lru_eviction:
            stats["lru_evictions_total"] = self._eviction_count
            stats["lru_tracked_seqs"] = len(self._last_access)
        if self._swap_enabled:
            stats["swapped_seqs"] = len(self._swapped_seqs)
            stats["cpu_free_blocks"] = len(self._cpu_free_blocks)
            stats["swap_count_total"] = self._swap_count
        return stats


class _nullcontext:
    """Minimal context manager for when no CUDA stream is needed."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
