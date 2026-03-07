"""
vllm-i64 :: Search History — Partitioned by API key

Token-routed isolation for search history:
each user identity is deterministically routed to its own partition.
No cross-partition access is possible by design.

    partition = sha256(api_key ∥ user_id) mod N

Even if a key leaks, it can only see its own history.
Persistence: optional JSON-per-partition on disk.
User controls deletion via DELETE /v1/search/history.

INL - 2025
"""

import json
import time
import hashlib
import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("vllm_i64.search.history")

NUM_PARTITIONS = 64


@dataclass
class SearchEntry:
    query: str
    sources: List[dict]
    answer: str
    timestamp: float = field(default_factory=time.time)


class SearchPartition:
    """Single isolated partition — holds history for one set of routed keys."""

    __slots__ = ("entries", "lock")

    def __init__(self) -> None:
        self.entries: Dict[str, List[SearchEntry]] = {}
        self.lock = threading.Lock()

    def add(self, identity: str, entry: SearchEntry, max_per_key: int = 200) -> None:
        with self.lock:
            if identity not in self.entries:
                self.entries[identity] = []
            history = self.entries[identity]
            history.append(entry)
            if len(history) > max_per_key:
                self.entries[identity] = history[-max_per_key:]

    def get(self, identity: str, limit: int = 50) -> List[SearchEntry]:
        with self.lock:
            return list(self.entries.get(identity, []))[-limit:]

    def clear(self, identity: str) -> int:
        with self.lock:
            return len(self.entries.pop(identity, []))

    def to_dict(self) -> Dict[str, list]:
        """Serialize for persistence."""
        with self.lock:
            return {
                k: [asdict(e) for e in v]
                for k, v in self.entries.items()
            }

    def load_dict(self, data: Dict[str, list]) -> None:
        """Restore from serialized data."""
        with self.lock:
            for identity, entries in data.items():
                self.entries[identity] = [
                    SearchEntry(**e) for e in entries
                ]


class PartitionedSearchHistory:
    """
    Token-routed search history with strict partition isolation.

    Each user identity is routed to exactly one partition via:
        partition_id = sha256(api_key ∥ user_id) mod N

    Properties:
        - Deterministic: same identity always routes to same partition
        - Isolated: no cross-partition data access
        - Persistent: optional save/load to disk (partitioned JSON files)
        - User-controlled: deletion via API, no forced TTL
    """

    def __init__(
        self,
        num_partitions: int = NUM_PARTITIONS,
        max_per_key: int = 200,
        persist_dir: Optional[str] = None,
    ):
        self.num_partitions = num_partitions
        self.max_per_key = max_per_key
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._partitions = [SearchPartition() for _ in range(num_partitions)]

        if self._persist_dir:
            self._load_from_disk()

        logger.info(
            "Search history: %d partitions, max=%d/key, persist=%s",
            num_partitions, max_per_key,
            str(self._persist_dir) if self._persist_dir else "off",
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
            return
        identity = self._identity(api_key, user_id)
        entry = SearchEntry(query=query, sources=sources, answer=answer)
        self._partition(identity).add(identity, entry, self.max_per_key)
        logger.debug("Recorded search for %s in partition %d", identity[:16], self._route(identity))

        if self._persist_dir:
            self._save_partition(self._route(identity))

    def get_history(
        self, api_key: str, limit: int = 50, user_id: Optional[str] = None,
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
        partition_idx = self._route(identity)
        removed = self._partition(identity).clear(identity)

        if self._persist_dir and removed:
            self._save_partition(partition_idx)

        return removed

    # ── Persistence ───────────────────────────────────────────────────────

    def _partition_path(self, idx: int) -> Path:
        return self._persist_dir / f"partition_{idx:03d}.json"

    def _save_partition(self, idx: int) -> None:
        """Save a single partition to disk."""
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            data = self._partitions[idx].to_dict()
            if not data:
                # Remove empty partition file
                path = self._partition_path(idx)
                if path.exists():
                    path.unlink()
                return
            with open(self._partition_path(idx), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to save partition %d: %s", idx, e)

    def _load_from_disk(self) -> None:
        """Load all partition files from disk."""
        if not self._persist_dir or not self._persist_dir.exists():
            return
        loaded = 0
        for idx in range(self.num_partitions):
            path = self._partition_path(idx)
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self._partitions[idx].load_dict(data)
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load partition %d: %s", idx, e)
        if loaded:
            logger.info("Loaded %d search history partitions from %s", loaded, self._persist_dir)

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
            "max_per_key": self.max_per_key,
            "persist_dir": str(self._persist_dir) if self._persist_dir else None,
        }
