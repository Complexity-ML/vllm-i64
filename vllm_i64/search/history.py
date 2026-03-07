"""
vllm-i64 :: Search History — Partitioned by API key

Token-routed isolation for search history:
each API key is deterministically routed to its own partition.
No cross-partition access is possible by design.

    partition = hash(api_key) % num_partitions

Even if a key leaks, it can only see its own history.
History is stored in-memory with TTL eviction — never persisted to disk.

INL - 2025
"""

import time
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("vllm_i64.search.history")

# Number of isolated partitions (power of 2 for fast modulo)
NUM_PARTITIONS = 64


@dataclass
class SearchEntry:
    query: str
    sources: List[dict]    # [{"index": 1, "title": ..., "url": ...}, ...]
    answer: str
    timestamp: float = field(default_factory=time.time)


class SearchPartition:
    """Single isolated partition — holds history for one set of routed keys."""

    __slots__ = ("entries", "lock")

    def __init__(self) -> None:
        self.entries: Dict[str, List[SearchEntry]] = {}  # api_key → entries
        self.lock = threading.Lock()

    def add(self, api_key: str, entry: SearchEntry, max_per_key: int = 50) -> None:
        with self.lock:
            if api_key not in self.entries:
                self.entries[api_key] = []
            history = self.entries[api_key]
            history.append(entry)
            # Cap history per key
            if len(history) > max_per_key:
                self.entries[api_key] = history[-max_per_key:]

    def get(self, api_key: str, limit: int = 20) -> List[SearchEntry]:
        with self.lock:
            return list(self.entries.get(api_key, []))[-limit:]

    def clear(self, api_key: str) -> int:
        with self.lock:
            removed = len(self.entries.pop(api_key, []))
            return removed

    def evict_expired(self, ttl_seconds: float) -> int:
        """Remove entries older than TTL. Returns count of evicted entries."""
        cutoff = time.time() - ttl_seconds
        evicted = 0
        with self.lock:
            dead_keys = []
            for key, entries in self.entries.items():
                before = len(entries)
                entries[:] = [e for e in entries if e.timestamp > cutoff]
                evicted += before - len(entries)
                if not entries:
                    dead_keys.append(key)
            for key in dead_keys:
                del self.entries[key]
        return evicted


class PartitionedSearchHistory:
    """
    Token-routed search history with strict partition isolation.

    Each API key is routed to exactly one partition via:
        partition_id = hash(api_key) % NUM_PARTITIONS

    Properties:
        - Deterministic: same key always routes to same partition
        - Isolated: no cross-partition data access
        - In-memory only: never persisted (no disk leakage)
        - TTL eviction: old entries auto-expire
    """

    def __init__(
        self,
        num_partitions: int = NUM_PARTITIONS,
        ttl_seconds: float = 3600,      # 1 hour default
        max_per_key: int = 50,
    ):
        self.num_partitions = num_partitions
        self.ttl_seconds = ttl_seconds
        self.max_per_key = max_per_key
        self._partitions = [SearchPartition() for _ in range(num_partitions)]
        logger.info(
            "Search history: %d partitions, TTL=%ds, max=%d/key",
            num_partitions, int(ttl_seconds), max_per_key,
        )

    def _identity(self, api_key: str, user_id: Optional[str] = None) -> str:
        """
        Build isolation identity from api_key + optional user_id.

        Team API key scenario:
            api_key="team-abc", user_id="alice" → "team-abc::alice"
            api_key="team-abc", user_id="bob"   → "team-abc::bob"
            → Different partitions, different history. No cross-user leakage.

        Solo API key scenario:
            api_key="personal-xyz", user_id=None → "personal-xyz"
        """
        if user_id:
            return f"{api_key}::{user_id}"
        return api_key

    def _route(self, identity: str) -> int:
        """Deterministic routing: identity → partition index."""
        h = hashlib.sha256(identity.encode()).digest()
        return int.from_bytes(h[:4], "big") % self.num_partitions

    def _partition(self, identity: str) -> SearchPartition:
        return self._partitions[self._route(identity)]

    def record(
        self, api_key: str, query: str, sources: List[dict], answer: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Record a search interaction in the user's routed partition."""
        if not api_key:
            return  # anonymous requests are not tracked
        identity = self._identity(api_key, user_id)
        entry = SearchEntry(query=query, sources=sources, answer=answer)
        self._partition(identity).add(identity, entry, self.max_per_key)
        logger.debug("Recorded search for %s in partition %d", identity[:16], self._route(identity))

    def get_history(
        self, api_key: str, limit: int = 20, user_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Get search history for a user (only from their own partition).
        Returns list of {"query", "sources", "answer", "timestamp"}.
        """
        if not api_key:
            return []
        identity = self._identity(api_key, user_id)
        entries = self._partition(identity).get(identity, limit)
        return [
            {
                "query": e.query,
                "sources": e.sources,
                "answer": e.answer,
                "timestamp": e.timestamp,
            }
            for e in entries
        ]

    def clear_history(self, api_key: str, user_id: Optional[str] = None) -> int:
        """Clear all search history for a user. Returns count removed."""
        if not api_key:
            return 0
        identity = self._identity(api_key, user_id)
        return self._partition(identity).clear(identity)

    def evict_all(self) -> int:
        """Run TTL eviction across all partitions."""
        total = 0
        for p in self._partitions:
            total += p.evict_expired(self.ttl_seconds)
        if total:
            logger.info("Evicted %d expired search entries", total)
        return total

    def stats(self) -> dict:
        """Aggregate stats across all partitions."""
        total_keys = 0
        total_entries = 0
        for p in self._partitions:
            with p.lock:
                total_keys += len(p.entries)
                total_entries += sum(len(v) for v in p.entries.values())
        return {
            "num_partitions": self.num_partitions,
            "total_keys": total_keys,
            "total_entries": total_entries,
            "ttl_seconds": self.ttl_seconds,
            "max_per_key": self.max_per_key,
        }
