"""
vllm-i64 :: API Middleware

CORS, authentication, rate limiting, and load shedding middleware.

INL - 2025
"""

import asyncio
import hmac
import time
from typing import Optional, Dict

from aiohttp import web

from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.middleware")


class TokenBucketRateLimiter:
    """Per-IP token bucket rate limiter with automatic stale bucket cleanup."""

    def __init__(self, requests_per_minute: int, max_buckets: int = 10000, cleanup_interval: float = 300.0):
        self.rate = requests_per_minute / 60.0
        self.capacity = requests_per_minute
        self._buckets: Dict[str, list] = {}  # ip -> [tokens, last_time]
        self._max_buckets = max_buckets
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.monotonic()
        self._lock = asyncio.Lock()

    async def allow(self, ip: str) -> bool:
        async with self._lock:
            return self._allow_unlocked(ip)

    def _allow_unlocked(self, ip: str) -> bool:
        """Token bucket check — must be called while holding self._lock."""
        now = time.monotonic()

        # Periodic cleanup of stale buckets to prevent memory leaks
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_stale(now)

        bucket = self._buckets.get(ip)
        if bucket is None:
            # Evict oldest bucket if at capacity
            if len(self._buckets) >= self._max_buckets:
                self._cleanup_stale(now)
                # If still at capacity after cleanup, evict oldest
                if len(self._buckets) >= self._max_buckets:
                    oldest_ip = min(self._buckets, key=lambda k: self._buckets[k][1])
                    del self._buckets[oldest_ip]

            self._buckets[ip] = [self.capacity - 1.0, now]
            return True

        tokens, last = bucket
        elapsed = now - last
        tokens = min(self.capacity, tokens + elapsed * self.rate)
        if tokens >= 1.0:
            bucket[0] = tokens - 1.0
            bucket[1] = now
            return True
        bucket[0] = tokens
        bucket[1] = now
        return False

    def _cleanup_stale(self, now: float):
        """Remove buckets that haven't been accessed recently (fully replenished)."""
        stale_threshold = self.capacity / self.rate  # Time to fully replenish
        stale_ips = [
            ip for ip, (_, last) in self._buckets.items()
            if now - last > stale_threshold
        ]
        for ip in stale_ips:
            del self._buckets[ip]
        self._last_cleanup = now


def make_cors_middleware():
    """Create CORS middleware that adds headers to all responses."""
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp
    return cors_middleware


def make_auth_middleware(api_key: str):
    """Create authentication middleware using constant-time comparison."""
    @web.middleware
    async def auth_middleware(request, handler):
        if request.path.startswith("/v1/"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return web.json_response(
                    {"error": {"message": "Invalid API key", "type": "authentication_error"}},
                    status=401,
                )
            provided_key = auth[7:]
            if not hmac.compare_digest(provided_key, api_key):
                return web.json_response(
                    {"error": {"message": "Invalid API key", "type": "authentication_error"}},
                    status=401,
                )
        return await handler(request)
    return auth_middleware


def make_rate_limit_middleware(rate_limiter: TokenBucketRateLimiter):
    """Create per-IP rate limiting middleware."""
    @web.middleware
    async def rate_limit_middleware(request, handler):
        if request.path.startswith("/v1/"):
            ip = request.remote or "unknown"
            if not await rate_limiter.allow(ip):
                return web.json_response(
                    {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                    status=429,
                    headers={"Retry-After": "60"},
                )
        return await handler(request)
    return rate_limit_middleware


def make_load_shed_middleware(get_load_fn, max_pending: int):
    """Create load shedding middleware (503 when queue is full)."""
    @web.middleware
    async def load_shed_middleware(request, handler):
        if request.path.startswith("/v1/"):
            current_load = get_load_fn()
            if current_load >= max_pending:
                return web.json_response(
                    {"error": {"message": "Server overloaded, try again later", "type": "overloaded_error"}},
                    status=503,
                    headers={"Retry-After": "5"},
                )
        return await handler(request)
    return load_shed_middleware
