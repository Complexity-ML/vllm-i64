"""
vllm-i64 :: Block Pool

Copied and adapted from vLLM (Apache 2.0).
Source: vllm/v1/core/kv_cache_utils.py + vllm/v1/core/block_pool.py

Provides:
  - KVCacheBlock      : lightweight block metadata with ref-counting
  - FreeKVCacheBlockQueue : O(1) doubly-linked list for free blocks (LRU order)
  - BlockPool         : allocate/free/prefix-cache blocks

Stripped of all vLLM-specific dependencies (vllm.distributed, Request, etc.)
INL - 2025
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# KVCacheBlock
# ---------------------------------------------------------------------------

@dataclass
class KVCacheBlock:
    """KV-cache block metadata (ref-counted, hash-tagged for prefix cache)."""

    block_id: int
    ref_cnt: int = 0
    # SHA-256 hash of this block's token content — set when block is full.
    block_hash: Optional[bytes] = None

    # Doubly-linked list pointers (managed by FreeKVCacheBlockQueue only).
    prev_free_block: Optional["KVCacheBlock"] = field(default=None, repr=False)
    next_free_block: Optional["KVCacheBlock"] = field(default=None, repr=False)

    # Null block never gets cached (used as a placeholder).
    is_null: bool = False

    def reset_hash(self) -> None:
        self.block_hash = None

    def __repr__(self) -> str:
        prev = self.prev_free_block.block_id if self.prev_free_block else None
        nxt  = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(id={self.block_id}, ref={self.ref_cnt}, "
            f"hash={self.block_hash and self.block_hash.hex()[:8]!r}, "
            f"prev={prev}, next={nxt})"
        )


# ---------------------------------------------------------------------------
# FreeKVCacheBlockQueue  (O(1) doubly-linked list — from vLLM)
# ---------------------------------------------------------------------------

class FreeKVCacheBlockQueue:
    """
    Doubly-linked list of free KVCacheBlocks.

    Supports O(1) popleft, append, and remove-from-middle.
    LRU order: most-recently-freed blocks go to the tail (evicted last).

    Adapted from vLLM (Apache 2.0).
    """

    def __init__(self, blocks: List[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Wire up consecutive blocks.
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Sentinel head and tail (never popped).
        self._head = KVCacheBlock(block_id=-1)
        self._tail = KVCacheBlock(block_id=-2)

        if blocks:
            self._head.next_free_block = blocks[0]
            blocks[0].prev_free_block  = self._head
            self._tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self._tail
        else:
            self._head.next_free_block = self._tail
            self._tail.prev_free_block = self._head

    # ------------------------------------------------------------------
    def popleft(self) -> KVCacheBlock:
        """Remove and return the LRU (front) block. O(1)."""
        if self.num_free_blocks == 0:
            raise RuntimeError("No free blocks in the pool")
        block = self._head.next_free_block
        assert block is not None and block is not self._tail
        self._remove(block)
        return block

    def popleft_n(self, n: int) -> List[KVCacheBlock]:
        """Remove and return n LRU blocks. O(n)."""
        if n > self.num_free_blocks:
            raise RuntimeError(
                f"Requested {n} blocks but only {self.num_free_blocks} free"
            )
        return [self.popleft() for _ in range(n)]

    def append(self, block: KVCacheBlock) -> None:
        """Append block to the tail (MRU position = evicted last). O(1)."""
        prev = self._tail.prev_free_block
        assert prev is not None
        prev.next_free_block   = block
        block.prev_free_block  = prev
        block.next_free_block  = self._tail
        self._tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks: List[KVCacheBlock]) -> None:
        for b in blocks:
            self.append(b)

    def remove(self, block: KVCacheBlock) -> None:
        """Remove block from an arbitrary position. O(1)."""
        self._remove(block)

    # ------------------------------------------------------------------
    def _remove(self, block: KVCacheBlock) -> None:
        prev = block.prev_free_block
        nxt  = block.next_free_block
        assert prev is not None and nxt is not None
        prev.next_free_block = nxt
        nxt.prev_free_block  = prev
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1


# ---------------------------------------------------------------------------
# BlockPool
# ---------------------------------------------------------------------------

class BlockPool:
    """
    Block allocator with LRU eviction and prefix caching.

    Adapted from vLLM's BlockPool (Apache 2.0).
    Removed: vllm.distributed, Request coupling, KVCacheEvent queue.
    Added  : standalone SHA-256 prefix-cache hashing.

    Usage:
        pool = BlockPool(num_gpu_blocks=512, enable_caching=True, block_size=16)
        blocks = pool.get_new_blocks(4)
        pool.cache_block(block, token_ids)
        pool.free_blocks(blocks)
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        block_size: int,
    ) -> None:
        assert num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.block_size = block_size

        # All blocks.
        self.blocks: List[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)
        ]

        # Free list (LRU doubly-linked list).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Prefix cache: hash → KVCacheBlock.
        # Only populated when enable_caching=True.
        self._hash_to_block: Dict[bytes, KVCacheBlock] = {}

        # Null block — block 0 is reserved as a no-op placeholder.
        self.null_block: KVCacheBlock = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """
        Allocate num_blocks from the free queue (LRU eviction).
        Evicts prefix-cached blocks as needed.
        """
        if num_blocks > self.get_num_free_blocks():
            raise RuntimeError(
                f"OOM: need {num_blocks} free blocks, "
                f"have {self.get_num_free_blocks()}"
            )
        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            if self.enable_caching:
                self._evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt = 1
        return blocks

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Increment ref-count (hit via prefix cache — remove from free list)."""
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, blocks: Iterable[KVCacheBlock]) -> None:
        """
        Decrement ref-count. Blocks reaching 0 go back to the free list tail
        (MRU position → evicted last → good for prefix cache reuse).
        Pass blocks in reverse order so the one closest to the prompt tail
        goes back last (= stays longest in cache).
        """
        released = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                released.append(block)
        self.free_block_queue.append_n(released)

    # ------------------------------------------------------------------
    # Prefix caching
    # ------------------------------------------------------------------

    @staticmethod
    def hash_block(token_ids: List[int], prev_hash: Optional[bytes] = None) -> bytes:
        """SHA-256 hash of a token block, chained from prev_hash."""
        h = hashlib.sha256()
        if prev_hash is not None:
            h.update(prev_hash)
        for tid in token_ids:
            h.update(tid.to_bytes(8, "little", signed=True))
        return h.digest()

    def get_cached_block(self, block_hash: bytes) -> Optional[KVCacheBlock]:
        """Look up a block by hash. Returns None on miss."""
        if not self.enable_caching:
            return None
        return self._hash_to_block.get(block_hash)

    def cache_block(self, block: KVCacheBlock, block_hash: bytes) -> None:
        """Register a full block in the prefix cache."""
        if not self.enable_caching or block.is_null:
            return
        if block.block_hash is None:
            block.block_hash = block_hash
            self._hash_to_block[block_hash] = block

    def _evict_cached_block(self, block: KVCacheBlock) -> None:
        """Remove a block from the prefix cache when it's being reallocated."""
        if block.block_hash is not None:
            self._hash_to_block.pop(block.block_hash, None)
            block.reset_hash()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Fraction of blocks in use (0.0 – 1.0). Null block excluded."""
        total = self.num_gpu_blocks - 1  # exclude null block
        if not total:
            return 0.0
        return 1.0 - self.get_num_free_blocks() / total
