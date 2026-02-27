"""
vllm-i64 :: Prometheus Metrics

Monitoring for production deployments.
All counters are integer (consistent with i64 philosophy).

Metrics:
  - vllm_i64_requests_total: total requests served
  - vllm_i64_tokens_generated_total: total tokens generated
  - vllm_i64_tokens_prompt_total: total prompt tokens processed
  - vllm_i64_request_duration_seconds: request latency histogram
  - vllm_i64_batch_size: current batch size gauge
  - vllm_i64_kv_cache_usage: KV cache block usage ratio
  - vllm_i64_experts_load: tokens per expert (load balance)

INL - 2025
"""

import time
from typing import Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class I64Metrics:
    """Prometheus metrics for vllm-i64."""

    def __init__(self, port: int = 9090, model_name: str = ""):
        self.enabled = HAS_PROMETHEUS
        if not self.enabled:
            return

        # Info
        self.model_info = Info("vllm_i64_model", "Model information")
        self.model_info.info({"name": model_name, "engine": "vllm-i64"})

        # Counters (integer)
        self.requests_total = Counter(
            "vllm_i64_requests_total", "Total requests served"
        )
        self.tokens_generated = Counter(
            "vllm_i64_tokens_generated_total", "Total tokens generated"
        )
        self.tokens_prompt = Counter(
            "vllm_i64_tokens_prompt_total", "Total prompt tokens processed"
        )

        # Histograms
        self.request_duration = Histogram(
            "vllm_i64_request_duration_seconds",
            "Request latency",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.time_per_token = Histogram(
            "vllm_i64_time_per_token_seconds",
            "Time per output token",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        )

        # Gauges
        self.batch_size = Gauge(
            "vllm_i64_batch_size", "Current batch size"
        )
        self.kv_cache_usage = Gauge(
            "vllm_i64_kv_cache_usage_ratio", "KV cache block usage (0-1)"
        )
        self.pending_requests = Gauge(
            "vllm_i64_pending_requests", "Requests waiting in queue"
        )

        # Start metrics server
        start_http_server(port)

    def on_request_start(self):
        """Called when a new request starts."""
        if not self.enabled:
            return time.perf_counter()
        self.requests_total.inc()
        return time.perf_counter()

    def on_request_end(self, start_time: float, prompt_tokens: int, output_tokens: int):
        """Called when a request completes."""
        if not self.enabled:
            return
        elapsed = time.perf_counter() - start_time
        self.request_duration.observe(elapsed)
        self.tokens_generated.inc(output_tokens)
        self.tokens_prompt.inc(prompt_tokens)
        if output_tokens > 0:
            self.time_per_token.observe(elapsed / output_tokens)

    def update_batch_size(self, size: int):
        if self.enabled:
            self.batch_size.set(size)

    def update_kv_usage(self, used: int, total: int):
        if self.enabled and total > 0:
            self.kv_cache_usage.set(used / total)

    def update_pending(self, count: int):
        if self.enabled:
            self.pending_requests.set(count)
