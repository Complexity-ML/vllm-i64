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
        if kv_cache_dtype == "fp8" and device != "cpu":
            self.kv_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2" and device != "cpu":
            self.kv_dtype = torch.float8_e5m2
        else:
            self.kv_dtype = dtype
        self.dtype = dtype  # backward compat (read_kv returns this dtype)

        # Derive from a reasonable max sequence length (2048 default, configurable)
        max_blocks_per_seq = max(128, (num_blocks * block_size) // block_size)
        self.max_blocks_per_seq = max_blocks_per_seq

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
        # Kept on CPU — it's integer metadata, no GPU compute needed.
        # Avoids GPU→CPU sync on every block lookup.
        self.block_table = torch.full(
            (max_seqs, max_blocks_per_seq), -1, dtype=torch.int32, device="cpu"
        )

        # Free block list (integer)
        self.free_blocks: List[int] = list(range(num_blocks))

        # Sequence lengths (integer) — CPU for fast lookups without GPU sync
        self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32, device="cpu")

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

        # ── Lazy block zeroing ──
        # Blocks are zeroed on allocation, not on free. This makes
        # free_sequence() O(1) per block and defers the GPU write to
        # the next allocation — beneficial for bulk LRU evictions.
        self._dirty_blocks: Set[int] = set()

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

            # Count blocks held by this sequence (vectorized)
            victim_blocks = int((self.block_table[victim_sid] >= 0).sum().item())

            # Free it
            self.free_sequence(victim_sid)
            freed += victim_blocks
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
        current_blocks = (self.block_table[seq_id] >= 0).sum().item()

        for i in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            # Lazy zeroing: only zero blocks that were previously used
            if block_id in self._dirty_blocks:
                self._zero_block(block_id)
                self._dirty_blocks.discard(block_id)
            block_idx = current_blocks + i
            self.block_table[seq_id, block_idx] = block_id
            allocated.append(block_id)

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

        # Pre-allocate any missing blocks (single batch call)
        unique_blocks = sorted(set(int(b) for b in block_indices))
        missing = sum(1 for bidx in unique_blocks if self.block_table[seq_id, bidx].item() < 0)
        if missing > 0:
            self.allocate_blocks(seq_id, missing)

        # Fetch all physical block IDs in one tensor op (1 GPU→CPU transfer)
        block_map_tensor = self.block_table[seq_id, unique_blocks]
        block_map = {bidx: int(block_map_tensor[i]) for i, bidx in enumerate(unique_blocks)}

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
        self.seq_lens[seq_id] = max(self.seq_lens[seq_id].item(), max_pos)
        self._touch(seq_id)

    def get_cache_tensors(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw k/v cache tensors for a layer.
        Shape: (num_blocks, block_size, num_kv_heads, head_dim) — flash_attn compatible.
        """
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def get_block_table_for_seqs(self, seq_ids: List[int]) -> torch.Tensor:
        """Extract block table rows for given sequences. Returns (num_seqs, max_blocks_per_seq) int32."""
        return self.block_table[seq_ids]

    def get_cache_seqlens(self, seq_ids: List[int]) -> torch.Tensor:
        """Get current sequence lengths. Returns (num_seqs,) int32."""
        return self.seq_lens[seq_ids]

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
                if cached_tokens is not None and cached_tokens != tuple(block_tokens):
                    break  # Hash collision — don't reuse wrong KV blocks
                # Reuse: point this seq's block table to existing physical block
                physical_block = self._prefix_hash_to_blocks[prefix_hash][0]
                self.block_table[seq_id, block_idx] = physical_block
                self._prefix_refcount[prefix_hash] = self._prefix_refcount.get(prefix_hash, 1) + 1
                reused += self.block_size
            else:
                break  # Can't skip non-contiguous blocks

        if reused > 0:
            self.seq_lens[seq_id] = reused

        return reused

    def register_prefix_blocks(self, seq_id: int, token_ids: List[int]):
        """
        After computing a full prefill, register the resulting KV blocks
        as reusable prefix blocks for future requests.
        """
        if not self.prefix_cache_enabled:
            return

        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            prefix_hash = self._hash_token_block(block_tokens)

            physical_block = self.block_table[seq_id, block_idx].item()
            if physical_block >= 0 and prefix_hash not in self._prefix_hash_to_blocks:
                self._prefix_hash_to_blocks[prefix_hash] = [physical_block]
                self._block_to_prefix[physical_block] = prefix_hash
                self._prefix_refcount[prefix_hash] = 1
                self._prefix_hash_to_tokens[prefix_hash] = tuple(block_tokens)

    def _zero_block(self, block_id: int):
        """Zero out K/V data in a physical block to prevent stale data leaking."""
        for layer_idx in range(self.num_layers):
            self.k_caches[layer_idx][block_id].zero_()
            self.v_caches[layer_idx][block_id].zero_()

    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence. Respects prefix cache refcounts."""
        blocks = self.block_table[seq_id]
        # Fetch all block IDs in one GPU→CPU transfer
        block_ids = blocks.tolist()
        for i, block_id in enumerate(block_ids):
            if block_id >= 0:
                # Check if this block is a shared prefix block
                prefix_hash = getattr(self, "_block_to_prefix", {}).get(block_id)
                if prefix_hash is not None:
                    rc = self._prefix_refcount.get(prefix_hash, 1) - 1
                    if rc <= 0:
                        # Last user — free the block (lazy zero on next alloc)
                        self._prefix_hash_to_blocks.pop(prefix_hash, None)
                        self._block_to_prefix.pop(block_id, None)
                        self._prefix_refcount.pop(prefix_hash, None)
                        self._dirty_blocks.add(block_id)
                        self.free_blocks.append(block_id)
                    else:
                        self._prefix_refcount[prefix_hash] = rc
                else:
                    self._dirty_blocks.add(block_id)
                    self.free_blocks.append(block_id)
                blocks[i] = -1

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
            (self.max_seqs, 128), -1, dtype=torch.int32, device="cpu"
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

        # Count GPU blocks held (one GPU→CPU transfer)
        blocks = self.block_table[seq_id]
        block_list = blocks.tolist()
        num_blocks = 0
        for bid in block_list:
            if bid >= 0:
                num_blocks += 1
            else:
                break

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
        }
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
