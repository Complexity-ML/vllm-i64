"""
vllm-i64 :: API Tracking Utilities

Usage tracking, latency tracking, request logging, priority management, and request caching.

INL - 2025
"""

import time
import json
import hashlib
import logging
from typing import Optional, Dict, List
from collections import OrderedDict, deque


class UsageTracker:
    """Per-API-key token usage tracking."""

    def __init__(self):
        self._usage: Dict[str, dict] = {}  # key → {prompt_tokens, completion_tokens, requests}

    def record(self, api_key: str, prompt_tokens: int, completion_tokens: int):
        if api_key not in self._usage:
            self._usage[api_key] = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        self._usage[api_key]["prompt_tokens"] += prompt_tokens
        self._usage[api_key]["completion_tokens"] += completion_tokens
        self._usage[api_key]["requests"] += 1

    def get(self, api_key: Optional[str] = None) -> dict:
        if api_key:
            return self._usage.get(api_key, {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0})
        return dict(self._usage)

    def get_total(self) -> dict:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        for v in self._usage.values():
            total["prompt_tokens"] += v["prompt_tokens"]
            total["completion_tokens"] += v["completion_tokens"]
            total["requests"] += v["requests"]
        return total


class RequestCache:
    """
    Request deduplication cache using OrderedDict for O(1) eviction.
    Caches generation results by prompt fingerprint to avoid recomputing identical requests.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()  # fingerprint → (result_dict, timestamp)

    def _fingerprint(self, prompt: str, max_tokens: int, **sampling_kwargs) -> str:
        """Create a cache key from request params. Only cache deterministic requests (temp=0)."""
        temperature = sampling_kwargs.get("temperature", 0.0)
        if temperature > 0:
            return ""  # Don't cache non-deterministic requests
        # Include ALL sampling params to prevent wrong cache hits
        parts = [prompt, str(max_tokens)]
        for k in sorted(sampling_kwargs.keys()):
            parts.append(f"{k}={sampling_kwargs[k]}")
        key = "|".join(parts)
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, prompt: str, max_tokens: int, **sampling_kwargs) -> Optional[dict]:
        fp = self._fingerprint(prompt, max_tokens, **sampling_kwargs)
        if not fp or fp not in self._cache:
            return None
        result, ts = self._cache[fp]
        if time.monotonic() - ts > self.ttl:
            del self._cache[fp]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(fp)
        return result

    def put(self, prompt: str, max_tokens: int, result: dict, **sampling_kwargs):
        fp = self._fingerprint(prompt, max_tokens, **sampling_kwargs)
        if not fp:
            return
        # Evict oldest (first item) if at capacity — O(1)
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[fp] = (result, time.monotonic())

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate_info(self) -> dict:
        return {"cached_entries": len(self._cache), "max_size": self.max_size}


class LatencyTracker:
    """Track request latencies for percentile computation (p50/p95/p99)."""

    def __init__(self, max_window: int = 1000):
        self.max_window = max_window
        self._latencies: deque = deque(maxlen=max_window)
        self._per_endpoint: Dict[str, deque] = {}

    def record(self, endpoint: str, latency_ms: float):
        """Record a request latency."""
        self._latencies.append(latency_ms)
        if endpoint not in self._per_endpoint:
            self._per_endpoint[endpoint] = deque(maxlen=self.max_window)
        self._per_endpoint[endpoint].append(latency_ms)

    def percentiles(self, endpoint: Optional[str] = None) -> Dict[str, float]:
        """Compute p50/p95/p99 from recent latencies."""
        data = list(self._per_endpoint.get(endpoint, [])) if endpoint else list(self._latencies)
        if not data:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0}
        data.sort()
        n = len(data)
        return {
            "p50_ms": round(data[int(n * 0.50)], 2),
            "p95_ms": round(data[min(int(n * 0.95), n - 1)], 2),
            "p99_ms": round(data[min(int(n * 0.99), n - 1)], 2),
            "count": n,
            "avg_ms": round(sum(data) / n, 2),
        }

    def get_all_endpoints(self) -> Dict[str, Dict[str, float]]:
        """Get percentiles for all endpoints."""
        result = {"overall": self.percentiles()}
        for ep in self._per_endpoint:
            result[ep] = self.percentiles(ep)
        return result


class RequestLogger:
    """Structured JSON request logging."""

    def __init__(self, enabled: bool = True, max_log: int = 10000):
        self.enabled = enabled
        self._log: deque = deque(maxlen=max_log)
        self._json_logger = logging.getLogger("vllm_i64.requests")

    def log_request(
        self,
        endpoint: str,
        status: int,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        api_key: Optional[str] = None,
        error: Optional[str] = None,
        request_id: Optional[str] = None,
        partition: Optional[int] = None,
    ):
        if not self.enabled:
            return
        entry = {
            "ts": time.time(),
            "endpoint": endpoint,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "api_key": api_key[:8] + "..." if api_key and len(api_key) > 8 else api_key,
        }
        if partition is not None:
            entry["partition"] = partition
        if request_id:
            entry["request_id"] = request_id
        if error:
            entry["error"] = error
        self._log.append(entry)
        self._json_logger.info(json.dumps(entry))

    def get_recent(self, n: int = 50) -> List[dict]:
        """Get last N log entries."""
        return list(self._log)[-n:]


class PriorityManager:
    """API key priority levels for request scheduling."""

    def __init__(self):
        self._priorities: Dict[str, int] = {}  # api_key → priority (higher = sooner)

    def set_priority(self, api_key: str, priority: int):
        self._priorities[api_key] = priority

    def get_priority(self, api_key: Optional[str], request_priority: int = 0) -> int:
        """Get effective priority: max of key-level and request-level."""
        key_prio = self._priorities.get(api_key, 0) if api_key else 0
        return max(key_prio, request_priority)

    def get_all(self) -> Dict[str, int]:
        return dict(self._priorities)
