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
import hashlib
from typing import Optional, Tuple, List, Dict, Set


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
        self.device = device

        # FP8 quantization: store KV in FP8 for 2x memory savings
        if kv_cache_dtype == "fp8" and device != "cpu":
            self.kv_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2" and device != "cpu":
            self.kv_dtype = torch.float8_e5m2
        else:
            self.kv_dtype = dtype
        self.dtype = dtype  # backward compat (read_kv returns this dtype)

        max_blocks_per_seq = 128  # max sequence length / block_size

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
        self.block_table = torch.full(
            (max_seqs, max_blocks_per_seq), -1, dtype=torch.int32, device=device
        )

        # Free block list (integer)
        self.free_blocks: List[int] = list(range(num_blocks))

        # Sequence lengths (integer)
        self.seq_lens = torch.zeros(max_seqs, dtype=torch.int32, device=device)

        # ── LRU eviction tracking (all integer) ──
        self.enable_lru_eviction = enable_lru_eviction
        self._access_counter: int = 0          # monotonic i64 clock
        self._last_access: Dict[int, int] = {} # seq_id → last access tick
        self._eviction_count: int = 0          # total evictions performed
        self._evicted_seq_ids: List[int] = []  # recently evicted (for scheduler)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def _touch(self, seq_id: int):
        """Update LRU access tick for a sequence. Pure integer."""
        self._access_counter += 1
        self._last_access[seq_id] = self._access_counter

    def _evict_lru(self, num_blocks_needed: int, protect_seq_id: int = -1) -> int:
        """
        Evict least recently used sequences until we have enough free blocks.

        Args:
            num_blocks_needed: how many blocks we need to free
            protect_seq_id: don't evict this sequence (the one requesting blocks)

        Returns:
            Number of blocks freed.
        """
        freed = 0
        self._evicted_seq_ids.clear()

        # Find active sequences (seq_lens > 0), sorted by LRU (oldest access first)
        active_seqs = []
        for sid in range(self.max_seqs):
            if self.seq_lens[sid].item() > 0 and sid != protect_seq_id:
                tick = self._last_access.get(sid, 0)
                active_seqs.append((tick, sid))

        # Sort by access tick ascending — least recently used first
        active_seqs.sort()

        for _tick, victim_sid in active_seqs:
            if freed >= num_blocks_needed:
                break

            # Count blocks held by this sequence
            blocks = self.block_table[victim_sid]
            victim_blocks = 0
            for i in range(blocks.shape[0]):
                if blocks[i].item() >= 0:
                    victim_blocks += 1

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
            block_idx = current_blocks + i
            self.block_table[seq_id, block_idx] = block_id
            allocated.append(block_id)

        self._touch(seq_id)
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

        self.k_caches[layer_idx][physical_block, offset, :, :] = k.to(self.kv_dtype)
        self.v_caches[layer_idx][physical_block, offset, :, :] = v.to(self.kv_dtype)
        self.seq_lens[seq_id] = max(self.seq_lens[seq_id].item(), position + 1)
        self._touch(seq_id)

    def read_kv(
        self,
        layer_idx: int,
        seq_id: int,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read all K/V for a sequence.

        Returns:
            k: (seq_len, num_kv_heads, head_dim)
            v: (seq_len, num_kv_heads, head_dim)
        """
        self._touch(seq_id)
        seq_len = self.seq_lens[seq_id].item()
        if max_len is not None:
            seq_len = min(seq_len, max_len)

        # Always return in compute dtype (dequantize if FP8)
        k_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.device)
        v_out = torch.zeros(seq_len, self.num_kv_heads, self.head_dim, dtype=self.compute_dtype, device=self.device)

        for pos in range(seq_len):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            physical_block = self.block_table[seq_id, block_idx].item()

            if physical_block >= 0:
                k_out[pos] = self.k_caches[layer_idx][physical_block, offset, :, :].to(self.compute_dtype)
                v_out[pos] = self.v_caches[layer_idx][physical_block, offset, :, :].to(self.compute_dtype)

        return k_out, v_out

    def write_kv_batch(
        self,
        layer_idx: int,
        seq_id: int,
        positions: torch.Tensor,   # (n,) int32
        k: torch.Tensor,           # (n, num_kv_heads, head_dim)
        v: torch.Tensor,           # (n, num_kv_heads, head_dim)
    ):
        """Batch write K/V for one sequence. Avoids per-token Python loop overhead."""
        for t in range(positions.shape[0]):
            pos = positions[t].item()
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            physical_block = self.block_table[seq_id, block_idx].item()

            if physical_block < 0:
                [physical_block] = self.allocate_blocks(seq_id, 1)

            self.k_caches[layer_idx][physical_block, offset, :, :] = k[t].to(self.kv_dtype)
            self.v_caches[layer_idx][physical_block, offset, :, :] = v[t].to(self.kv_dtype)

        max_pos = positions.max().item() + 1
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

    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence. Respects prefix cache refcounts."""
        blocks = self.block_table[seq_id]
        for i in range(blocks.shape[0]):
            block_id = blocks[i].item()
            if block_id >= 0:
                # Check if this block is a shared prefix block
                prefix_hash = getattr(self, "_block_to_prefix", {}).get(block_id)
                if prefix_hash is not None:
                    rc = self._prefix_refcount.get(prefix_hash, 1) - 1
                    if rc <= 0:
                        # Last user — free the block
                        self._prefix_hash_to_blocks.pop(prefix_hash, None)
                        self._block_to_prefix.pop(block_id, None)
                        self._prefix_refcount.pop(prefix_hash, None)
                        self.free_blocks.append(block_id)
                    else:
                        self._prefix_refcount[prefix_hash] = rc
                else:
                    self.free_blocks.append(block_id)
                blocks[i] = -1

        self.seq_lens[seq_id] = 0
        self._last_access.pop(seq_id, None)

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
        return stats
