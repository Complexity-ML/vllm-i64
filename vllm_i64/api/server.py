"""
vllm-i64 :: API Server (aiohttp)

OpenAI-compatible API for token-routed inference.
text → tokenize → i64 → async engine → i64 → detokenize → text

Uses AsyncI64Engine for continuous batching:
  - Multiple concurrent requests are batched together
  - Each forward pass processes mixed prefill + decode
  - Maximum GPU utilization and tok/s

Endpoints:
    POST /v1/completions     → completion (sync + streaming)
    POST /v1/chat/completions → chat completion (sync + streaming)
    GET  /health             → health check + engine stats
    GET  /v1/models          → list models

INL - 2025
"""

import json
import time
import asyncio
import signal
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass, asdict

from aiohttp import web

from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine, GenerationResult
from vllm_i64.core.sampling import SamplingParams
from vllm_i64.core.logits_processor import OutputConstraints
from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.server")


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stream: bool = False
    # Structured output
    response_format: Optional[Dict] = None  # {"type": "json_object"} or {"type": "regex", "pattern": "..."}
    stop: Optional[list] = None  # Stop sequences (strings)
    # Beam search
    n: int = 1
    best_of: int = 1
    # Logprobs
    logprobs: Optional[int] = None  # Number of top logprobs per token

    def validate(self, max_seq_len: int = 2048) -> Optional[str]:
        """Validate request parameters. Returns error message or None."""
        if not self.prompt or not self.prompt.strip():
            return "prompt must not be empty"
        if self.max_tokens < 1:
            return "max_tokens must be >= 1"
        if self.max_tokens > max_seq_len:
            return f"max_tokens must be <= {max_seq_len}"
        if self.temperature < 0:
            return "temperature must be >= 0"
        if self.top_k < 0:
            return "top_k must be >= 0"
        if self.top_p < 0 or self.top_p > 1:
            return "top_p must be in [0, 1]"
        if self.repetition_penalty <= 0:
            return "repetition_penalty must be > 0"
        if self.logprobs is not None and (self.logprobs < 0 or self.logprobs > 20):
            return "logprobs must be between 0 and 20"
        return None

    def to_sampling_params(self) -> SamplingParams:
        """Convert API request to engine sampling params."""
        # Build output constraints from response_format and stop sequences
        constraints = None
        has_constraints = self.response_format or self.stop
        if has_constraints:
            stop_seqs = None
            if self.stop:
                stop_seqs = [[int(b) for b in s.encode("utf-8")] for s in self.stop]
            constraints = OutputConstraints(
                json_mode=bool(self.response_format and self.response_format.get("type") == "json_object"),
                regex_pattern=(
                    self.response_format.get("pattern")
                    if self.response_format and self.response_format.get("type") == "regex"
                    else None
                ),
                stop_sequences=stop_seqs,
            )

        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            json_mode=bool(self.response_format and self.response_format.get("type") == "json_object"),
            num_beams=self.best_of if self.best_of > 1 else 1,
            logprobs=self.logprobs,
            output_constraints=constraints,
        )


@dataclass
class CompletionResponse:
    id: str
    object: str = "text_completion"
    created: int = 0
    model: str = "inl-token-routed"
    choices: List[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


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
    Request deduplication cache.
    Caches generation results by prompt fingerprint to avoid recomputing identical requests.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # fingerprint → (result_dict, timestamp)

    def _fingerprint(self, prompt: str, temperature: float, top_k: int, top_p: float, max_tokens: int) -> str:
        """Create a cache key from request params. Only cache deterministic requests (temp=0)."""
        import hashlib
        if temperature > 0:
            return ""  # Don't cache non-deterministic requests
        key = f"{prompt}|{temperature}|{top_k}|{top_p}|{max_tokens}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, prompt: str, temperature: float, top_k: int, top_p: float, max_tokens: int) -> Optional[dict]:
        fp = self._fingerprint(prompt, temperature, top_k, top_p, max_tokens)
        if not fp or fp not in self._cache:
            return None
        result, ts = self._cache[fp]
        if time.monotonic() - ts > self.ttl:
            del self._cache[fp]
            return None
        return result

    def put(self, prompt: str, temperature: float, top_k: int, top_p: float, max_tokens: int, result: dict):
        fp = self._fingerprint(prompt, temperature, top_k, top_p, max_tokens)
        if not fp:
            return
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[fp] = (result, time.monotonic())

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate_info(self) -> dict:
        return {"cached_entries": len(self._cache), "max_size": self.max_size}


class TokenBucketRateLimiter:
    """Per-IP token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute / 60.0
        self.capacity = requests_per_minute
        self._buckets: Dict[str, list] = {}  # ip -> [tokens, last_time]

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.get(ip)
        if bucket is None:
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


class I64Server:
    """
    Inference server with async continuous batching.

    Uses AsyncI64Engine: multiple HTTP requests are automatically
    batched together for maximum throughput.

    All internal operations are integer.
    String handling is only at the API boundary.
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
    ):
        # Wrap sync engine in async engine for continuous batching
        self.async_engine = AsyncI64Engine.__new__(AsyncI64Engine)
        self.async_engine.engine = engine
        self.async_engine._pending_futures = {}
        self.async_engine._request_times = {}
        self.async_engine._engine_task = None
        self.async_engine._running = False
        self.async_engine._draining = False
        self.async_engine.active_requests = 0
        self.async_engine.peak_batch_size = 0

        self.sync_engine = engine
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.model_name = model_name
        self.host = host
        self.port = port
        self.request_counter: int = 0
        self.api_key = api_key
        self._rate_limiter = TokenBucketRateLimiter(rate_limit) if rate_limit > 0 else None
        self._max_pending = max_pending
        self._usage_tracker = UsageTracker()
        self._request_cache = RequestCache()
        self._start_time = time.monotonic()

    def _tokenize(self, text: str) -> List[int]:
        """Text → i64 token IDs. Boundary operation."""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return [int(b) for b in text.encode("utf-8")]

    def _detokenize(self, token_ids: List[int]) -> str:
        """i64 token IDs → text. Boundary operation."""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        return bytes(token_ids).decode("utf-8", errors="replace")

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template to messages → prompt string."""
        if self.chat_template:
            from jinja2 import Template
            tmpl = Template(self.chat_template)
            prompt = tmpl.render(messages=messages, add_generation_prompt=True)
            # Ensure generation prompt is appended (template may not handle add_generation_prompt)
            if not prompt.rstrip().endswith("Assistant:"):
                prompt = prompt.rstrip("\n") + "\nAssistant:"
            return prompt
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _build_response(self, result: GenerationResult, prompt_ids: List[int]) -> CompletionResponse:
        """Build completion response from generation result."""
        output_text = self._detokenize(result.output_tokens)
        self.request_counter += 1

        choice = {
            "text": output_text,
            "index": 0,
            "finish_reason": result.finish_reason,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(result.output_tokens),
                "total_tokens": len(prompt_ids) + len(result.output_tokens),
                "engine_steps": result.num_steps,
                "elapsed_ms": round(result.elapsed_ms, 2),
            },
        }

        # Include logprobs if available
        if result.token_logprobs:
            choice["logprobs"] = {
                "tokens": [self._detokenize([lp.token_id]) for lp in result.token_logprobs],
                "token_logprobs": [lp.logprob for lp in result.token_logprobs],
                "top_logprobs": [lp.top_logprobs for lp in result.token_logprobs],
            }

        return CompletionResponse(
            id=f"cmpl-{self.request_counter}",
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
        )

    async def _async_complete(self, request: CompletionRequest, api_key: Optional[str] = None) -> CompletionResponse:
        """Async completion — batched with other concurrent requests."""
        prompt_ids = self._tokenize(request.prompt)

        result = await self.async_engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(),
        )

        resp = self._build_response(result, prompt_ids)

        # Track usage
        if api_key:
            self._usage_tracker.record(api_key, len(prompt_ids), len(result.output_tokens))

        return resp

    async def _async_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming completion via async engine — OpenAI SSE format."""
        prompt_ids = self._tokenize(request.prompt)
        self.request_counter += 1
        stream_id = f"cmpl-{self.request_counter}"
        created = int(time.time())

        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(),
        ):
            token_text = self._detokenize([token_id])
            chunk = {
                "id": stream_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "text": token_text,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        final = {
            "id": stream_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": "length",
            }],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    # =====================================================================
    # aiohttp handlers
    # =====================================================================

    async def handle_completions(self, request: web.Request) -> web.Response:
        """POST /v1/completions — async continuous batching."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        prompt = body.get("prompt")
        if not prompt:
            return web.json_response(
                {"error": {"message": "Missing 'prompt' field", "type": "invalid_request_error"}},
                status=400,
            )

        # Wrap raw prompt in chat template if model is chat-trained
        if self.chat_template:
            prompt = self._apply_chat_template([
                {"role": "user", "content": prompt},
            ])

        req = CompletionRequest(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 0.8),
            top_k=body.get("top_k", 50),
            top_p=body.get("top_p", 0.9),
            repetition_penalty=body.get("repetition_penalty", 1.1),
            stream=body.get("stream", False),
            response_format=body.get("response_format"),
            stop=body.get("stop"),
            n=body.get("n", 1),
            best_of=body.get("best_of", 1),
            logprobs=body.get("logprobs"),
        )

        # Validate request parameters
        error = req.validate(max_seq_len=self.sync_engine.scheduler.max_seq_len)
        if error:
            return web.json_response(
                {"error": {"message": error, "type": "invalid_request_error"}},
                status=400,
            )

        # Extract API key for usage tracking
        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None

        try:
            if req.stream:
                response = web.StreamResponse()
                response.content_type = "text/event-stream"
                await response.prepare(request)
                try:
                    async for chunk in self._async_stream(req):
                        await response.write(chunk.encode())
                except (ConnectionResetError, ConnectionError):
                    pass  # client disconnected
                return response

            # Check dedup cache
            cached = self._request_cache.get(req.prompt, req.temperature, req.top_k, req.top_p, req.max_tokens)
            if cached is not None:
                return web.json_response(cached)

            result = await self._async_complete(req, api_key=req_api_key)
            result_dict = result.to_dict()

            # Store in dedup cache
            self._request_cache.put(req.prompt, req.temperature, req.top_k, req.top_p, req.max_tokens, result_dict)

            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            pass  # client disconnected
        except Exception as e:
            logger.error(f"Completion error: {e}", exc_info=True)
            return web.json_response(
                {"error": {"message": str(e), "type": "server_error"}},
                status=500,
            )

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """POST /v1/chat/completions — async continuous batching."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        messages = body.get("messages")
        if not messages:
            return web.json_response(
                {"error": {"message": "Missing 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )
        prompt = self._apply_chat_template(messages)

        req = CompletionRequest(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 0.8),
            top_k=body.get("top_k", 50),
            top_p=body.get("top_p", 0.9),
            repetition_penalty=body.get("repetition_penalty", 1.1),
            stream=body.get("stream", False),
        )

        try:
            if req.stream:
                response = web.StreamResponse()
                response.content_type = "text/event-stream"
                await response.prepare(request)
                try:
                    async for chunk in self._async_stream(req):
                        await response.write(chunk.encode())
                except (ConnectionResetError, ConnectionError):
                    pass  # client disconnected
                return response

            result = await self._async_complete(req)
            result_dict = result.to_dict()
            if result_dict["choices"]:
                text = result_dict["choices"][0]["text"]
                finish_reason = result_dict["choices"][0].get("finish_reason", "length")
                message = {"role": "assistant", "content": text}

                # Parse tool calls if tools were provided
                tools = body.get("tools")
                if tools:
                    from vllm_i64.core.tool_parser import ToolCallParser
                    parser = ToolCallParser(tools)
                    tool_calls = parser.parse(text)
                    if tool_calls:
                        message["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": tc.function_arguments,
                                },
                            }
                            for tc in tool_calls
                        ]
                        finish_reason = "tool_calls"

                result_dict["choices"][0] = {
                    "message": message,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            result_dict["object"] = "chat.completion"
            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            pass  # client disconnected during non-stream
        except Exception as e:
            logger.error(f"Chat completion error: {e}", exc_info=True)
            return web.json_response(
                {"error": {"message": str(e), "type": "server_error"}},
                status=500,
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health — detailed health check with KV cache, GPU memory, queue depth."""
        stats = self.async_engine.get_stats()
        uptime_s = int(time.monotonic() - self._start_time)
        health = {
            "status": "ok",
            "model": self.model_name,
            "uptime_seconds": uptime_s,
            "requests_served": self.request_counter,
            "engine": stats,
            "queue": {
                "pending": len(self.sync_engine.scheduler.pending),
                "running": len(self.sync_engine.scheduler.running),
                "active_requests": self.async_engine.active_requests,
            },
            "cache": self._request_cache.hit_rate_info,
            "usage": self._usage_tracker.get_total(),
        }
        # GPU memory info
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                health["gpu"] = {
                    "free_mb": round(mem[0] / 1e6),
                    "total_mb": round(mem[1] / 1e6),
                    "used_mb": round((mem[1] - mem[0]) / 1e6),
                    "utilization_pct": round((1 - mem[0] / mem[1]) * 100, 1),
                }
        except Exception:
            pass
        # KV cache stats
        if self.sync_engine.kv_cache is not None:
            health["kv_cache"] = self.sync_engine.kv_cache.get_stats()
        return web.json_response(health)

    async def handle_models(self, request: web.Request) -> web.Response:
        """GET /v1/models"""
        return web.json_response({
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "owned_by": "inl",
            }],
        })

    async def handle_tokenize(self, request: web.Request) -> web.Response:
        """POST /v1/tokenize — tokenize text, return token IDs and count."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        text = body.get("text")
        messages = body.get("messages")

        if messages:
            text = self._apply_chat_template(messages)
        elif not text:
            return web.json_response(
                {"error": {"message": "Missing 'text' or 'messages'", "type": "invalid_request_error"}},
                status=400,
            )

        tokens = self._tokenize(text)
        return web.json_response({"tokens": tokens, "count": len(tokens)})

    async def handle_embeddings(self, request: web.Request) -> web.Response:
        """POST /v1/embeddings — compute embeddings for input text."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        input_data = body.get("input")
        if not input_data:
            return web.json_response(
                {"error": {"message": "Missing 'input' field", "type": "invalid_request_error"}},
                status=400,
            )

        # Normalize to list
        if isinstance(input_data, str):
            input_data = [input_data]

        try:
            embeddings = []
            for i, text in enumerate(input_data):
                token_ids = self._tokenize(text)
                embedding = self.sync_engine.embed(token_ids)
                embeddings.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                })

            total_tokens = sum(len(self._tokenize(t)) for t in input_data)
            return web.json_response({
                "object": "list",
                "data": embeddings,
                "model": self.model_name,
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            })
        except Exception as e:
            return web.json_response(
                {"error": {"message": str(e), "type": "server_error"}},
                status=500,
            )

    async def handle_usage(self, request: web.Request) -> web.Response:
        """GET /v1/usage — token usage stats per API key."""
        api_key = None
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            api_key = auth[7:]

        if api_key:
            usage = self._usage_tracker.get(api_key)
        else:
            usage = self._usage_tracker.get_total()

        return web.json_response({"usage": usage})

    async def handle_lora_load(self, request: web.Request) -> web.Response:
        """POST /v1/lora/load — load a LoRA adapter."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        adapter_id = body.get("adapter_id")
        adapter_name = body.get("name", f"adapter-{adapter_id}")
        adapter_path = body.get("path")
        scaling = body.get("scaling", 1.0)

        if adapter_id is None or adapter_path is None:
            return web.json_response(
                {"error": {"message": "Missing 'adapter_id' or 'path'", "type": "invalid_request_error"}},
                status=400,
            )

        # Enable LoRA on first load
        if self.sync_engine._lora_manager is None:
            try:
                self.sync_engine.enable_lora()
            except Exception as e:
                return web.json_response(
                    {"error": {"message": f"Failed to enable LoRA: {e}", "type": "server_error"}},
                    status=500,
                )

        success = self.sync_engine.load_lora_adapter(int(adapter_id), adapter_name, adapter_path, scaling)
        if success:
            return web.json_response({"status": "ok", "adapter_id": adapter_id, "name": adapter_name})
        return web.json_response(
            {"error": {"message": "Failed to load adapter (no weights found)", "type": "server_error"}},
            status=500,
        )

    async def handle_lora_unload(self, request: web.Request) -> web.Response:
        """POST /v1/lora/unload — unload a LoRA adapter."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        adapter_id = body.get("adapter_id")
        if adapter_id is None:
            return web.json_response(
                {"error": {"message": "Missing 'adapter_id'", "type": "invalid_request_error"}},
                status=400,
            )

        self.sync_engine.unload_lora_adapter(int(adapter_id))
        return web.json_response({"status": "ok", "adapter_id": adapter_id})

    async def handle_lora_list(self, request: web.Request) -> web.Response:
        """GET /v1/lora/list — list loaded LoRA adapters."""
        adapters = self.sync_engine.list_lora_adapters()
        return web.json_response({
            "adapters": [{"id": aid, "name": name} for aid, name in adapters.items()],
        })

    @web.middleware
    async def cors_middleware(self, request, handler):
        """Add CORS headers to all responses."""
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @web.middleware
    async def auth_middleware(self, request, handler):
        """Check Bearer token on /v1/* endpoints."""
        if self.api_key and request.path.startswith("/v1/"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != self.api_key:
                return web.json_response(
                    {"error": {"message": "Invalid API key", "type": "authentication_error"}},
                    status=401,
                )
        return await handler(request)

    @web.middleware
    async def rate_limit_middleware(self, request, handler):
        """Per-IP rate limiting on /v1/* endpoints."""
        if self._rate_limiter and request.path.startswith("/v1/"):
            ip = request.remote or "unknown"
            if not self._rate_limiter.allow(ip):
                return web.json_response(
                    {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                    status=429,
                    headers={"Retry-After": "60"},
                )
        return await handler(request)

    @web.middleware
    async def load_shed_middleware(self, request, handler):
        """Reject requests when queue is full (503 Service Unavailable)."""
        if self._max_pending > 0 and request.path.startswith("/v1/"):
            pending = len(self.sync_engine.scheduler.pending)
            active = self.async_engine.active_requests
            if pending + active >= self._max_pending:
                return web.json_response(
                    {"error": {"message": "Server overloaded, try again later", "type": "overloaded_error"}},
                    status=503,
                    headers={"Retry-After": "5"},
                )
        return await handler(request)

    def create_app(self) -> web.Application:
        """Create aiohttp application with routes and engine lifecycle."""
        middlewares = [self.cors_middleware]
        if self.api_key:
            middlewares.append(self.auth_middleware)
        if self._rate_limiter:
            middlewares.append(self.rate_limit_middleware)
        if self._max_pending > 0:
            middlewares.append(self.load_shed_middleware)
        app = web.Application(middlewares=middlewares)
        app.router.add_route("OPTIONS", "/v1/completions", self._handle_options)
        app.router.add_route("OPTIONS", "/v1/chat/completions", self._handle_options)
        app.router.add_post("/v1/completions", self.handle_completions)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/tokenize", self.handle_tokenize)
        app.router.add_post("/v1/embeddings", self.handle_embeddings)
        app.router.add_get("/v1/usage", self.handle_usage)
        app.router.add_post("/v1/lora/load", self.handle_lora_load)
        app.router.add_post("/v1/lora/unload", self.handle_lora_unload)
        app.router.add_get("/v1/lora/list", self.handle_lora_list)
        app.router.add_get("/", self.handle_root)

        # Start/stop async engine with the app
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def handle_root(self, request):
        """Redirect to demo page."""
        raise web.HTTPFound("https://complexity-website.vercel.app/demo")

    async def _handle_options(self, request):
        """Handle CORS preflight requests."""
        return web.Response()

    async def _on_startup(self, app):
        """Start the async engine loop when server starts."""
        await self.async_engine.start()
        logger.info("Engine started: continuous batching active")

    async def _on_cleanup(self, app):
        """Graceful shutdown: drain active requests, then stop engine."""
        logger.info("Server cleanup: draining requests...")
        await self.async_engine.stop(drain_timeout=30.0)
        logger.info("Server cleanup complete")

    def run(self):
        """Start the server with async continuous batching and graceful shutdown."""
        logger.info(f"vllm-i64 :: {self.model_name}")
        logger.info(f"  http://{self.host}:{self.port}")
        logger.info(f"  POST /v1/completions | POST /v1/chat/completions | GET /health")
        logger.info(f"  mode: async continuous batching")
        app = self.create_app()

        # Graceful shutdown on SIGTERM/SIGINT
        async def _shutdown_handler(app):
            logger.info("Received shutdown signal")

        app.on_shutdown.append(_shutdown_handler)
        web.run_app(app, host=self.host, port=self.port, print=None)
