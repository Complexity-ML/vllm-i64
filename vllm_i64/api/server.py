"""
vllm-i64 :: API Server (aiohttp)

OpenAI-compatible API for token-routed inference.

Endpoints:
    POST /v1/completions          → completion (sync + streaming)
    POST /v1/chat/completions      → chat completion (sync + streaming)
    GET  /health                  → health check
    GET  /v1/models               → list models
    GET  /v1/models/{id}          → model details
    POST /v1/tokenize             → tokenize text
    POST /v1/embeddings           → text embeddings
    GET  /v1/usage                → token usage
    POST /v1/batch                → batch completions
    GET  /v1/metrics              → latency + usage metrics
    GET  /v1/logs                 → recent request log
    POST /v1/priority             → set API key priority
    POST /v1/cancel/{id}          → cancel running request
    GET  /v1/ws/completions       → WebSocket streaming
    GET  /docs                    → OpenAPI spec
    POST /v1/lora/load            → load LoRA adapter
    POST /v1/lora/unload          → unload LoRA adapter
    GET  /v1/lora/list            → list loaded adapters
    GET  /v1/cache/stats          → KV cache statistics
    POST /v1/cache/purge          → purge prefix cache
    GET  /v1/monitor              → live monitoring snapshot
    GET  /v1/experts              → expert routing distribution
    POST /v1/rag/index            → index text for RAG
    POST /v1/rag/search           → search RAG index
    GET  /v1/rag/stats            → RAG statistics
    POST /v1/execute              → sandboxed code execution
    GET  /v1/agent/events         → SSE agent event stream
    GET  /v1/agent/history        → recent agent events

INL - 2025
"""

import asyncio
import concurrent.futures
import itertools
import time
from typing import Optional

from aiohttp import web

from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine
from vllm_i64.core.logging import get_logger
from vllm_i64.api.middleware import (
    TokenBucketRateLimiter,
    make_cors_middleware,
    make_auth_middleware,
    make_rate_limit_middleware,
    make_load_shed_middleware,
)
from vllm_i64.api.events import EventBus
from vllm_i64.api.tracking import UsageTracker, RequestCache, LatencyTracker, RequestLogger, PriorityManager
from vllm_i64.api.types import CompletionRequest, CompletionResponse  # noqa: F401 (re-export)
from vllm_i64.api._helpers import HelpersMixin
from vllm_i64.api._completions import CompletionsMixin
from vllm_i64.api._admin import AdminMixin
from vllm_i64.api._rag import RAGMixin
from vllm_i64.api._agent import AgentMixin

logger = get_logger("vllm_i64.server")


class I64Server(HelpersMixin, CompletionsMixin, AdminMixin, RAGMixin, AgentMixin):
    """
    Inference server with async continuous batching.

    Composed via mixins — each group of endpoints lives in its own module:
      _helpers.py      tokenize / detokenize / chat template / images / responses
      _completions.py  /v1/completions + /v1/chat/completions
      _admin.py        health / models / lora / batch / metrics / cache / monitor
      _rag.py          /v1/rag/*
      _agent.py        /v1/execute + /v1/agent/*
    """

    def __init__(
        self,
        engine: I64Engine,
        tokenizer=None,
        chat_template=None,
        model_name: str = "inl-token-routed",
        host: str = "0.0.0.0",
        port: int = 8000,
        api_key: Optional[str] = None,
        rate_limit: int = 0,
        max_pending: int = 0,
        rag_index_path: Optional[str] = None,
        sandbox_enabled: bool = False,
        sandbox_timeout: int = 30,
        sandbox_max_memory_mb: int = 256,
        sandbox_user: Optional[str] = None,
    ):
        # ── Engine ──────────────────────────────────────────────────────
        if engine is None:
            self.async_engine = None
            self.sync_engine = None
        else:
            from vllm_i64.cpu.engine import CPUEngine, AsyncCPUEngine
            if isinstance(engine, CPUEngine):
                self.async_engine = AsyncCPUEngine.from_cpu_engine(engine)
            else:
                self.async_engine = AsyncI64Engine.from_sync_engine(engine)
            self.sync_engine = engine

        self.tokenizer = tokenizer
        self.chat_template = chat_template

        # Fix EOS token if config disagrees with tokenizer
        if tokenizer and engine is not None and hasattr(engine, 'model') and engine.model is not None:
            tok_eos = tokenizer.eos_token_id
            cfg_eos = getattr(engine.model.config, 'eos_token_id', 0)
            if cfg_eos != tok_eos:
                logger.info("Fixing eos_token_id: config=%d → tokenizer=%d", cfg_eos, tok_eos)
                engine.model.config.eos_token_id = tok_eos

        # Token quality vector — pre-computed logit bias from tokenizer heuristics
        if tokenizer and engine is not None and hasattr(engine, 'sampler'):
            engine.sampler.set_token_quality_vector(tokenizer.token_quality_vector)

        # Space suppression at step 0 (first-token quality fix)
        self._space_suppress_ids = None
        if tokenizer:
            space_ids = tokenizer.encode(" ")
            if len(space_ids) == 1:
                self._space_suppress_ids = [space_ids[0]]
            elif len(space_ids) == 2 and space_ids[0] == tokenizer.bos_token_id:
                self._space_suppress_ids = [space_ids[1]]

        # ── Server metadata ──────────────────────────────────────────────
        self.model_name = model_name
        self.host = host
        self.port = port
        self._request_counter = itertools.count(1)
        self.request_counter: int = 0
        self.api_key = api_key
        self._rate_limiter = TokenBucketRateLimiter(rate_limit) if rate_limit > 0 else None
        self._max_pending = max_pending
        self._start_time = time.monotonic()
        self._shutting_down = False
        self._last_expert_response: dict | None = None

        # ── Trackers ─────────────────────────────────────────────────────
        self._usage_tracker = UsageTracker()
        self._request_cache = RequestCache()
        self._latency_tracker = LatencyTracker()
        self._request_logger = RequestLogger()
        self._priority_manager = PriorityManager()

        # ── Thread pool (tokenization off event loop) ─────────────────────
        self._tokenize_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="tokenize"
        )

        # ── RAG ──────────────────────────────────────────────────────────
        self.retriever = None
        self.rag_enabled = False
        self._rag_index_path = None
        if rag_index_path:
            try:
                import os
                from vllm_i64.rag import Retriever
                if os.path.exists(rag_index_path):
                    self.retriever = Retriever.load(rag_index_path)
                    logger.info("RAG enabled: loaded index from %s", rag_index_path)
                else:
                    self.retriever = Retriever()
                    logger.info("RAG enabled: empty index (will save to %s)", rag_index_path)
                self.rag_enabled = True
                self._rag_index_path = rag_index_path
            except Exception as e:
                logger.warning("RAG init failed: %s", e)
        else:
            try:
                from vllm_i64.rag import Retriever
                self.retriever = Retriever()
                self.rag_enabled = True
            except ImportError:
                pass

        # ── Sandbox ──────────────────────────────────────────────────────
        self.sandbox = None
        self.sandbox_enabled = sandbox_enabled
        if sandbox_enabled:
            from vllm_i64.sandbox import Sandbox
            self.sandbox = Sandbox(
                timeout=sandbox_timeout,
                max_memory_mb=sandbox_max_memory_mb,
                sandbox_user=sandbox_user,
            )
            level = "L2 (user: %s)" % sandbox_user if sandbox_user else "L1"
            logger.info("Sandbox enabled: %s, timeout=%ds memory=%dMB",
                        level, sandbox_timeout, sandbox_max_memory_mb)

        # ── Event bus ────────────────────────────────────────────────────
        self.event_bus = EventBus()

    # ======================================================================
    # App lifecycle
    # ======================================================================

    def create_app(self) -> web.Application:
        middlewares = [make_cors_middleware()]
        if self.api_key:
            middlewares.append(make_auth_middleware(self.api_key))
        if self._rate_limiter:
            middlewares.append(make_rate_limit_middleware(self._rate_limiter))
        if self._max_pending > 0 and self.sync_engine is not None:
            def _get_load():
                return len(self.sync_engine.scheduler.pending) + self.async_engine.active_requests
            middlewares.append(make_load_shed_middleware(_get_load, self._max_pending))

        app = web.Application(middlewares=middlewares, client_max_size=16 * 1024 * 1024)

        # Completions
        app.router.add_route("OPTIONS", "/v1/completions", self._handle_options)
        app.router.add_route("OPTIONS", "/v1/chat/completions", self._handle_options)
        app.router.add_post("/v1/completions", self.handle_completions)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)

        # Misc
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/v1/models/{model_id}", self.handle_model_info)
        app.router.add_post("/v1/tokenize", self.handle_tokenize)
        app.router.add_post("/v1/embeddings", self.handle_embeddings)
        app.router.add_get("/v1/usage", self.handle_usage)
        app.router.add_post("/v1/batch", self.handle_batch)
        app.router.add_get("/v1/metrics", self.handle_metrics)
        app.router.add_get("/v1/logs", self.handle_request_log)
        app.router.add_post("/v1/priority", self.handle_priority)
        app.router.add_post("/v1/cancel/{request_id}", self.handle_cancel)
        app.router.add_get("/v1/ws/completions", self.handle_ws_completions)
        app.router.add_get("/docs", self.handle_openapi)

        # LoRA
        app.router.add_post("/v1/lora/load", self.handle_lora_load)
        app.router.add_post("/v1/lora/unload", self.handle_lora_unload)
        app.router.add_get("/v1/lora/list", self.handle_lora_list)

        # Admin
        app.router.add_get("/v1/cache/stats", self.handle_cache_stats)
        app.router.add_post("/v1/cache/purge", self.handle_cache_purge)
        app.router.add_route("OPTIONS", "/v1/cache/purge", self._handle_options)
        app.router.add_get("/v1/monitor", self.handle_monitor)
        app.router.add_get("/v1/experts", self.handle_expert_stats)

        # RAG
        app.router.add_post("/v1/rag/index", self.handle_rag_index)
        app.router.add_post("/v1/rag/search", self.handle_rag_search)
        app.router.add_get("/v1/rag/stats", self.handle_rag_stats)
        app.router.add_route("OPTIONS", "/v1/rag/index", self._handle_options)
        app.router.add_route("OPTIONS", "/v1/rag/search", self._handle_options)

        # Agent / sandbox
        app.router.add_post("/v1/execute", self.handle_execute)
        app.router.add_route("OPTIONS", "/v1/execute", self._handle_options)
        app.router.add_get("/v1/agent/events", self.handle_agent_events)
        app.router.add_get("/v1/agent/history", self.handle_agent_history)
        app.router.add_route("OPTIONS", "/v1/agent/events", self._handle_options)

        app.router.add_get("/", self.handle_root)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def handle_root(self, request: web.Request) -> web.Response:
        raise web.HTTPFound("https://www.complexity-ai.fr/demo")

    async def _handle_options(self, request: web.Request) -> web.Response:
        return web.Response()

    async def _on_startup(self, app: web.Application) -> None:
        if self.async_engine is not None:
            await self.async_engine.start()
            logger.info("Engine started: continuous batching active")
        else:
            logger.info("Sandbox-only mode: no engine to start")

    async def _on_cleanup(self, app: web.Application) -> None:
        logger.info("Server cleanup: draining requests...")
        self._shutting_down = True
        if self.async_engine is not None:
            await self.async_engine.stop(drain_timeout=30.0)
        logger.info("Server cleanup complete")

    def run(self) -> None:
        logger.info("vllm-i64 :: %s", self.model_name)
        logger.info("  http://%s:%d", self.host, self.port)
        logger.info("  POST /v1/completions | POST /v1/chat/completions | GET /health")
        logger.info("  POST /v1/batch | GET /v1/models/{id} | GET /v1/metrics | GET /v1/logs")
        logger.info("  POST /v1/cancel/{id} | WS /v1/ws/completions | GET /docs")
        if self.rag_enabled:
            logger.info("  POST /v1/rag/index | POST /v1/rag/search | GET /v1/rag/stats")
        if self.sandbox_enabled:
            logger.info("  POST /v1/execute (sandbox: %ds timeout, %dMB memory)",
                        self.sandbox.timeout, self.sandbox.max_memory_mb)

        app = self.create_app()

        async def _graceful_shutdown(app: web.Application) -> None:
            logger.info("Graceful shutdown complete")

        app.on_shutdown.append(_graceful_shutdown)

        try:
            import signal as sig
            loop = asyncio.new_event_loop()
            loop.add_signal_handler(sig.SIGTERM, lambda: setattr(self, '_shutting_down', True))
        except (NotImplementedError, OSError):
            pass  # Windows

        try:
            web.run_app(app, host=self.host, port=self.port, print=None)
        except KeyboardInterrupt:
            pass
