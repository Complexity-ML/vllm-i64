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
import itertools
import concurrent.futures
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass, asdict

from aiohttp import web

from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine, GenerationResult
from vllm_i64.core.sampling import SamplingParams
from vllm_i64.core.logits_processor import OutputConstraints
from vllm_i64.core.logging import get_logger
from vllm_i64.api.middleware import (
    TokenBucketRateLimiter,
    make_cors_middleware,
    make_auth_middleware,
    make_rate_limit_middleware,
    make_load_shed_middleware,
)
from vllm_i64.api.tracking import (
    UsageTracker,
    RequestCache,
    LatencyTracker,
    RequestLogger,
    PriorityManager,
)

logger = get_logger("vllm_i64.server")


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0              # Min-p: dynamic threshold relative to top token
    typical_p: float = 1.0          # Typical sampling: entropy-based token selection
    repetition_penalty: float = 1.1
    min_tokens: int = 0             # Minimum tokens before allowing EOS
    stream: bool = False
    # Structured output
    response_format: Optional[Dict] = None  # {"type": "json_object"} or {"type": "regex", "pattern": "..."}
    stop: Optional[list] = None  # Stop sequences (strings)
    # Beam search
    n: int = 1
    best_of: int = 1
    # Logprobs
    logprobs: Optional[int] = None  # Number of top logprobs per token
    # Reproducibility
    seed: Optional[int] = None
    # Logit bias — {token_id_str: bias_float}
    logit_bias: Optional[Dict[str, float]] = None
    # Frequency/presence penalties (OpenAI-compatible)
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Priority (higher = sooner)
    priority: int = 0

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
        if self.min_p < 0 or self.min_p > 1:
            return "min_p must be in [0, 1]"
        if self.typical_p < 0 or self.typical_p > 1:
            return "typical_p must be in [0, 1]"
        if self.min_tokens < 0:
            return "min_tokens must be >= 0"
        if self.repetition_penalty <= 0:
            return "repetition_penalty must be > 0"
        if self.logprobs is not None and (self.logprobs < 0 or self.logprobs > 20):
            return "logprobs must be between 0 and 20"
        if self.frequency_penalty < -2.0 or self.frequency_penalty > 2.0:
            return "frequency_penalty must be in [-2.0, 2.0]"
        if self.presence_penalty < -2.0 or self.presence_penalty > 2.0:
            return "presence_penalty must be in [-2.0, 2.0]"
        if self.logit_bias:
            for k, v in self.logit_bias.items():
                if not k.lstrip('-').isdigit():
                    return f"logit_bias keys must be token ID strings, got '{k}'"
                if v < -100 or v > 100:
                    return f"logit_bias values must be in [-100, 100], got {v}"
        return None

    def to_sampling_params(self, tokenizer=None) -> SamplingParams:
        """Convert API request to engine sampling params."""
        # Build output constraints from response_format and stop sequences
        constraints = None
        has_constraints = self.response_format or self.stop
        if has_constraints:
            stop_seqs = None
            if self.stop and tokenizer is not None:
                # Encode stop sequences using the tokenizer for correct token-level matching
                stop_seqs = [tokenizer.encode(s) for s in self.stop]
            elif self.stop:
                # Fallback: byte-level encoding (works for byte-level tokenizers)
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

        # Convert logit_bias from {str: float} to {int: float}
        logit_bias = None
        if self.logit_bias:
            logit_bias = {int(k): v for k, v in self.logit_bias.items()}

        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            typical_p=self.typical_p,
            repetition_penalty=self.repetition_penalty,
            min_tokens=self.min_tokens,
            json_mode=bool(self.response_format and self.response_format.get("type") == "json_object"),
            num_beams=self.best_of if self.best_of > 1 else 1,
            logprobs=self.logprobs,
            output_constraints=constraints,
            seed=self.seed,
            logit_bias=logit_bias,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )


@dataclass
class CompletionResponse:
    id: str
    object: str = "text_completion"
    created: int = 0
    model: str = "inl-token-routed"
    choices: List[Dict] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []

    def to_dict(self) -> dict:
        d = asdict(self)
        # Attach top-level usage if present (OpenAI standard)
        if hasattr(self, '_usage'):
            d["usage"] = self._usage
        if hasattr(self, '_engine_metrics'):
            d["engine_metrics"] = self._engine_metrics
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


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
        # Create async engine — use dedicated CPU engine when on CPU
        from vllm_i64.cpu.engine import CPUEngine, AsyncCPUEngine
        if isinstance(engine, CPUEngine):
            self.async_engine = AsyncCPUEngine.from_cpu_engine(engine)
        else:
            self.async_engine = AsyncI64Engine.from_sync_engine(engine)

        self.sync_engine = engine
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.model_name = model_name
        self.host = host
        self.port = port
        # Thread-safe atomic request counter
        self._request_counter = itertools.count(1)
        self.request_counter: int = 0  # For stats display
        self.api_key = api_key
        self._rate_limiter = TokenBucketRateLimiter(rate_limit) if rate_limit > 0 else None
        self._max_pending = max_pending
        self._usage_tracker = UsageTracker()
        self._request_cache = RequestCache()
        self._start_time = time.monotonic()
        self._latency_tracker = LatencyTracker()
        self._request_logger = RequestLogger()
        self._priority_manager = PriorityManager()
        self._shutting_down = False
        # Thread pool for tokenization (avoids blocking the event loop)
        self._tokenize_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tokenize")
        # Single-token detokenize cache: token_id → text (avoids repeated decode calls)
        self._detok_cache: Dict[int, str] = {}

    def _next_request_id(self) -> str:
        """Generate a unique request ID (OpenAI-compatible format)."""
        import uuid
        n = next(self._request_counter)
        self.request_counter = n  # Update for stats
        return f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def _tokenize(self, text: str) -> List[int]:
        """Text → i64 token IDs. Boundary operation."""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return [int(b) for b in text.encode("utf-8")]

    async def _tokenize_async(self, text: str) -> List[int]:
        """Tokenize in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenize_pool, self._tokenize, text)

    def _detokenize(self, token_ids: List[int]) -> str:
        """i64 token IDs → text. Boundary operation."""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        # Byte fallback: only valid for byte-level models (vocab_size <= 256).
        # Mask to 8 bits to avoid ValueError when IDs exceed 255.
        safe_ids = [t & 0xFF for t in token_ids]
        return bytes(safe_ids).decode("utf-8", errors="replace")

    async def _detokenize_async(self, token_ids: List[int]) -> str:
        """Detokenize in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenize_pool, self._detokenize, token_ids)

    def _detokenize_token(self, token_id: int) -> str:
        """Single token → text with cache. For streaming."""
        text = self._detok_cache.get(token_id)
        if text is None:
            text = self._detokenize([token_id])
            if len(self._detok_cache) < 100000:  # Cap cache size
                self._detok_cache[token_id] = text
        return text

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template to messages → prompt string."""
        if self.chat_template:
            from jinja2 import Template
            tmpl = Template(self.chat_template)
            prompt = tmpl.render(messages=messages, add_generation_prompt=True)
            # Ensure the generation prompt is present (template may not handle add_generation_prompt)
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
        """Build completion response from generation result (OpenAI-compatible)."""
        output_text = self._detokenize(result.output_tokens)
        req_id = self._next_request_id()

        choice = {
            "text": output_text,
            "index": 0,
            "finish_reason": result.finish_reason,
        }

        # Include logprobs if available
        if result.token_logprobs:
            choice["logprobs"] = {
                "tokens": [self._detokenize([lp.token_id]) for lp in result.token_logprobs],
                "token_logprobs": [lp.logprob for lp in result.token_logprobs],
                "top_logprobs": [lp.top_logprobs for lp in result.token_logprobs],
            }

        resp = CompletionResponse(
            id=req_id,
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
        )

        # Attach top-level usage (OpenAI standard)
        resp._usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(result.output_tokens),
            "total_tokens": len(prompt_ids) + len(result.output_tokens),
        }
        # Engine-specific metrics (non-standard extension)
        resp._engine_metrics = {
            "engine_steps": result.num_steps,
            "elapsed_ms": round(result.elapsed_ms, 2),
        }
        return resp

    async def _async_complete(self, request: CompletionRequest, api_key: Optional[str] = None, endpoint: str = "/v1/completions") -> CompletionResponse:
        """Async completion — batched with other concurrent requests."""
        t0 = time.monotonic()
        prompt_ids = await self._tokenize_async(request.prompt)

        result = await self.async_engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
        )

        resp = self._build_response(result, prompt_ids)
        latency_ms = (time.monotonic() - t0) * 1000

        # Track usage
        if api_key:
            self._usage_tracker.record(api_key, len(prompt_ids), len(result.output_tokens))

        # Track latency
        self._latency_tracker.record(endpoint, latency_ms)

        # Log request
        self._request_logger.log_request(
            endpoint=endpoint, status=200, latency_ms=latency_ms,
            prompt_tokens=len(prompt_ids), completion_tokens=len(result.output_tokens),
            api_key=api_key,
        )

        return resp

    async def _async_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming completion via async engine — OpenAI SSE format."""
        prompt_ids = await self._tokenize_async(request.prompt)
        stream_id = self._next_request_id()
        created = int(time.time())

        last_token_id = None
        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
        ):
            last_token_id = token_id
            token_text = self._detokenize_token(token_id)
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

        # Detect finish reason from last token
        finish_reason = "length"
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        if last_token_id is None or (eos_id is not None and last_token_id == eos_id):
            finish_reason = "stop"

        final = {
            "id": stream_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": finish_reason,
            }],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    async def _async_chat_stream(
        self, request: CompletionRequest, tools: Optional[list] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion — OpenAI SSE format with delta objects."""
        prompt_ids = await self._tokenize_async(request.prompt)
        stream_id = self._next_request_id()
        created = int(time.time())

        # First chunk: role
        first_chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        # Content chunks
        last_token_id = None
        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
        ):
            last_token_id = token_id
            token_text = self._detokenize_token(token_id)
            chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Detect finish reason from last token
        finish_reason = "length"
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        if last_token_id is None or (eos_id is not None and last_token_id == eos_id):
            finish_reason = "stop"

        final = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
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
            min_p=body.get("min_p", 0.0),
            typical_p=body.get("typical_p", 1.0),
            repetition_penalty=body.get("repetition_penalty", 1.1),
            min_tokens=body.get("min_tokens", 0),
            stream=body.get("stream", False),
            response_format=body.get("response_format"),
            stop=body.get("stop"),
            n=body.get("n", 1),
            best_of=body.get("best_of", 1),
            logprobs=body.get("logprobs"),
            seed=body.get("seed"),
            logit_bias=body.get("logit_bias"),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            priority=body.get("priority", 0),
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
                response.headers["Cache-Control"] = "no-cache"
                await response.prepare(request)
                try:
                    async for chunk in self._async_stream(req):
                        await response.write(chunk.encode())
                        await response.drain()
                except (ConnectionResetError, ConnectionError):
                    pass  # client disconnected
                return response

            # Check dedup cache
            cache_kwargs = dict(
                temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                min_p=getattr(req, 'min_p', 0.0), typical_p=getattr(req, 'typical_p', 1.0),
                repetition_penalty=getattr(req, 'repetition_penalty', 1.0),
                frequency_penalty=getattr(req, 'frequency_penalty', 0.0),
                presence_penalty=getattr(req, 'presence_penalty', 0.0),
                seed=getattr(req, 'seed', None),
            )
            cached = self._request_cache.get(req.prompt, req.max_tokens, **cache_kwargs)
            if cached is not None:
                return web.json_response(cached)

            result = await self._async_complete(req, api_key=req_api_key)
            result_dict = result.to_dict()

            # Store in dedup cache
            self._request_cache.put(req.prompt, req.max_tokens, result_dict, **cache_kwargs)

            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            return web.Response(status=499, text="Client disconnected")
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
            min_p=body.get("min_p", 0.0),
            typical_p=body.get("typical_p", 1.0),
            repetition_penalty=body.get("repetition_penalty", 1.1),
            min_tokens=body.get("min_tokens", 0),
            stream=body.get("stream", False),
            response_format=body.get("response_format"),
            stop=body.get("stop"),
            n=body.get("n", 1),
            best_of=body.get("best_of", 1),
            logprobs=body.get("logprobs"),
            seed=body.get("seed"),
            logit_bias=body.get("logit_bias"),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            priority=body.get("priority", 0),
        )

        # Validate
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
                response.headers["Cache-Control"] = "no-cache"
                await response.prepare(request)
                try:
                    async for chunk in self._async_chat_stream(req, body.get("tools")):
                        await response.write(chunk.encode())
                        await response.drain()
                except (ConnectionResetError, ConnectionError):
                    pass  # client disconnected
                return response

            result = await self._async_complete(req, api_key=req_api_key, endpoint="/v1/chat/completions")
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
            return web.Response(status=499, text="Client disconnected")
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

        # Determine overall status
        status = "ok"
        checks = {}

        # Check model loaded
        model_loaded = self.sync_engine.model is not None
        checks["model_loaded"] = model_loaded

        # Check KV cache health
        if self.sync_engine.kv_cache is not None:
            kv_stats = self.sync_engine.kv_cache.get_stats()
            kv_usage = kv_stats["used_blocks"] / max(kv_stats["num_blocks"], 1)
            checks["kv_cache_usage_pct"] = round(kv_usage * 100, 1)
            if kv_usage > 0.95:
                status = "degraded"

        # Check CUDA health
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                gpu_info = {
                    "free_mb": round(mem[0] / 1e6),
                    "total_mb": round(mem[1] / 1e6),
                    "used_mb": round((mem[1] - mem[0]) / 1e6),
                    "utilization_pct": round((1 - mem[0] / mem[1]) * 100, 1),
                }
                checks["gpu"] = gpu_info
                if mem[0] / mem[1] < 0.05:  # Less than 5% free
                    status = "degraded"
            else:
                checks["gpu"] = "not_available"
        except Exception:
            checks["gpu"] = "error"

        health = {
            "status": status,
            "model": self.model_name,
            "uptime_seconds": uptime_s,
            "requests_served": self.request_counter,
            "engine": stats,
            "queue": {
                "pending": len(self.sync_engine.scheduler.pending),
                "running": len(self.sync_engine.scheduler.running),
                "active_requests": self.async_engine.active_requests,
            },
            "checks": checks,
            "cache": self._request_cache.hit_rate_info,
            "usage": self._usage_tracker.get_total(),
            "latency": self._latency_tracker.percentiles(),
        }
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
            total_tokens = 0
            for i, text in enumerate(input_data):
                token_ids = self._tokenize(text)
                total_tokens += len(token_ids)
                embedding = self.sync_engine.embed(token_ids)
                embeddings.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                })
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

    async def handle_batch(self, request: web.Request) -> web.Response:
        """POST /v1/batch — submit multiple prompts, return all results."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        requests_data = body.get("requests")
        if not requests_data or not isinstance(requests_data, list):
            return web.json_response(
                {"error": {"message": "Missing 'requests' array", "type": "invalid_request_error"}},
                status=400,
            )

        if len(requests_data) > 128:
            return web.json_response(
                {"error": {"message": "Max 128 requests per batch", "type": "invalid_request_error"}},
                status=400,
            )

        # Extract API key
        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None

        # Build all CompletionRequests
        completion_reqs = []
        for i, rd in enumerate(requests_data):
            prompt = rd.get("prompt")
            if not prompt:
                return web.json_response(
                    {"error": {"message": f"Request {i}: missing 'prompt'", "type": "invalid_request_error"}},
                    status=400,
                )
            req = CompletionRequest(
                prompt=prompt,
                max_tokens=rd.get("max_tokens", 256),
                temperature=rd.get("temperature", 0.8),
                top_k=rd.get("top_k", 50),
                top_p=rd.get("top_p", 0.9),
                min_p=rd.get("min_p", 0.0),
                typical_p=rd.get("typical_p", 1.0),
                repetition_penalty=rd.get("repetition_penalty", 1.1),
                min_tokens=rd.get("min_tokens", 0),
                seed=rd.get("seed"),
                logit_bias=rd.get("logit_bias"),
                frequency_penalty=rd.get("frequency_penalty", 0.0),
                presence_penalty=rd.get("presence_penalty", 0.0),
            )
            error = req.validate(max_seq_len=self.sync_engine.scheduler.max_seq_len)
            if error:
                return web.json_response(
                    {"error": {"message": f"Request {i}: {error}", "type": "invalid_request_error"}},
                    status=400,
                )
            completion_reqs.append(req)

        # Submit all concurrently via async engine
        tasks = [
            self._async_complete(req, api_key=req_api_key, endpoint="/v1/batch")
            for req in completion_reqs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build response
        responses = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                responses.append({"index": i, "error": str(r)})
            else:
                responses.append({"index": i, "result": r.to_dict()})

        return web.json_response({"responses": responses})

    async def handle_model_info(self, request: web.Request) -> web.Response:
        """GET /v1/models/{model_id} — detailed model information."""
        model_id = request.match_info.get("model_id", "")
        if model_id != self.model_name:
            return web.json_response(
                {"error": {"message": f"Model '{model_id}' not found", "type": "not_found_error"}},
                status=404,
            )

        info = {
            "id": self.model_name,
            "object": "model",
            "owned_by": "inl",
            "created": int(self._start_time),
        }

        # Add model config details if available
        if self.sync_engine.model is not None and hasattr(self.sync_engine.model, 'config'):
            config = self.sync_engine.model.config
            info["config"] = {
                "num_experts": getattr(config, 'num_experts', None),
                "vocab_size": getattr(config, 'vocab_size', None),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None),
                "num_key_value_heads": getattr(config, 'num_key_value_heads', None),
                "head_dim": getattr(config, 'head_dim', None),
            }
            # Count parameters
            total_params = sum(p.numel() for p in self.sync_engine.model.parameters())
            info["parameters"] = total_params
            info["dtype"] = str(next(self.sync_engine.model.parameters()).dtype)

        # Engine info
        info["engine"] = {
            "max_batch_size": self.sync_engine.scheduler.max_batch_size,
            "max_seq_len": self.sync_engine.scheduler.max_seq_len,
            "kv_cache": self.sync_engine.kv_cache is not None,
            "speculative": self.sync_engine.speculative_decoder is not None,
            "lora": self.sync_engine._lora_manager is not None,
        }

        return web.json_response(info)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """GET /v1/metrics — latency percentiles and request stats."""
        metrics = {
            "latency": self._latency_tracker.get_all_endpoints(),
            "usage": self._usage_tracker.get_total(),
            "cache": self._request_cache.hit_rate_info,
            "uptime_seconds": int(time.monotonic() - self._start_time),
            "requests_served": self.request_counter,
        }
        # Engine stats
        metrics["engine"] = self.async_engine.get_stats()
        return web.json_response(metrics)

    async def handle_request_log(self, request: web.Request) -> web.Response:
        """GET /v1/logs — recent request log entries."""
        n = int(request.query.get("n", "50"))
        entries = self._request_logger.get_recent(n)
        return web.json_response({"entries": entries, "count": len(entries)})

    async def handle_priority(self, request: web.Request) -> web.Response:
        """POST /v1/priority — set API key priority level."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )
        api_key = body.get("api_key")
        priority = body.get("priority", 0)
        if not api_key:
            return web.json_response(
                {"error": {"message": "Missing 'api_key'", "type": "invalid_request_error"}},
                status=400,
            )
        self._priority_manager.set_priority(api_key, int(priority))
        return web.json_response({"status": "ok", "api_key": api_key, "priority": priority})

    async def handle_cancel(self, request: web.Request) -> web.Response:
        """POST /v1/cancel/{request_id} — cancel a running request."""
        try:
            request_id = int(request.match_info["request_id"])
        except (KeyError, ValueError):
            return web.json_response(
                {"error": {"message": "Invalid request_id", "type": "invalid_request_error"}},
                status=400,
            )
        await self.async_engine.cancel_request(request_id)
        return web.json_response({"status": "ok", "cancelled": request_id})

    async def handle_ws_completions(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket /v1/ws/completions — streaming via WebSocket."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    body = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
                    continue

                prompt = body.get("prompt")
                if not prompt:
                    await ws.send_json({"error": {"message": "Missing 'prompt'", "type": "invalid_request_error"}})
                    continue

                req = CompletionRequest(
                    prompt=prompt,
                    max_tokens=body.get("max_tokens", 256),
                    temperature=body.get("temperature", 0.8),
                    top_k=body.get("top_k", 50),
                    top_p=body.get("top_p", 0.9),
                    stream=True,
                )

                error = req.validate(max_seq_len=self.sync_engine.scheduler.max_seq_len)
                if error:
                    await ws.send_json({"error": {"message": error, "type": "invalid_request_error"}})
                    continue

                stream_id = self._next_request_id()
                created = int(time.time())
                prompt_ids = self._tokenize(prompt)

                try:
                    last_token_id = None
                    async for token_id in self.async_engine.generate_stream(
                        prompt_token_ids=prompt_ids,
                        max_new_tokens=req.max_tokens,
                        sampling_params=req.to_sampling_params(tokenizer=self.tokenizer),
                    ):
                        last_token_id = token_id
                        token_text = self._detokenize_token(token_id)
                        await ws.send_json({
                            "id": stream_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [{"index": 0, "text": token_text, "finish_reason": None}],
                        })
                    ws_finish = "length"
                    eos_id = getattr(self.tokenizer, 'eos_token_id', None)
                    if last_token_id is None or (eos_id is not None and last_token_id == eos_id):
                        ws_finish = "stop"
                    await ws.send_json({
                        "id": stream_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [{"index": 0, "text": "", "finish_reason": ws_finish}],
                        "done": True,
                    })
                except Exception as e:
                    await ws.send_json({"error": {"message": str(e), "type": "server_error"}})
            elif msg.type == web.WSMsgType.ERROR:
                break

        return ws

    async def handle_openapi(self, request: web.Request) -> web.Response:
        """GET /docs — OpenAPI 3.0 spec."""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "vllm-i64 API",
                "version": "0.1.0",
                "description": "Integer-first inference engine for token-routed language models",
            },
            "paths": {
                "/v1/completions": {
                    "post": {
                        "summary": "Create completion",
                        "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/CompletionRequest"}}}},
                        "responses": {"200": {"description": "Completion response"}},
                    }
                },
                "/v1/chat/completions": {
                    "post": {
                        "summary": "Create chat completion",
                        "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"messages": {"type": "array"}, "max_tokens": {"type": "integer"}}}}}},
                        "responses": {"200": {"description": "Chat completion response"}},
                    }
                },
                "/v1/cancel/{request_id}": {
                    "post": {
                        "summary": "Cancel a running request",
                        "parameters": [{"name": "request_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                        "responses": {"200": {"description": "Cancellation confirmed"}},
                    }
                },
                "/v1/batch": {
                    "post": {
                        "summary": "Batch completions",
                        "responses": {"200": {"description": "Batch response"}},
                    }
                },
                "/v1/models": {
                    "get": {
                        "summary": "List models",
                        "responses": {"200": {"description": "Model list"}},
                    }
                },
                "/v1/ws/completions": {
                    "get": {
                        "summary": "WebSocket streaming completions",
                        "description": "Send JSON messages with prompt/max_tokens, receive streamed token chunks",
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check",
                        "responses": {"200": {"description": "Health status"}},
                    }
                },
                "/v1/metrics": {
                    "get": {
                        "summary": "Latency and usage metrics",
                        "responses": {"200": {"description": "Metrics data"}},
                    }
                },
            },
            "components": {
                "schemas": {
                    "CompletionRequest": {
                        "type": "object",
                        "required": ["prompt"],
                        "properties": {
                            "prompt": {"type": "string"},
                            "max_tokens": {"type": "integer", "default": 256},
                            "temperature": {"type": "number", "default": 0.8},
                            "top_k": {"type": "integer", "default": 50},
                            "top_p": {"type": "number", "default": 0.9},
                            "min_p": {"type": "number", "default": 0.0},
                            "typical_p": {"type": "number", "default": 1.0},
                            "repetition_penalty": {"type": "number", "default": 1.1},
                            "min_tokens": {"type": "integer", "default": 0},
                            "stream": {"type": "boolean", "default": False},
                            "stop": {"type": "array", "items": {"type": "string"}},
                            "seed": {"type": "integer"},
                            "logit_bias": {"type": "object"},
                            "frequency_penalty": {"type": "number", "default": 0.0},
                            "presence_penalty": {"type": "number", "default": 0.0},
                        },
                    },
                },
            },
        }
        return web.json_response(spec)

    def create_app(self) -> web.Application:
        """Create aiohttp application with routes and engine lifecycle."""
        middlewares = [make_cors_middleware()]
        if self.api_key:
            middlewares.append(make_auth_middleware(self.api_key))
        if self._rate_limiter:
            middlewares.append(make_rate_limit_middleware(self._rate_limiter))
        if self._max_pending > 0:
            def _get_load():
                return len(self.sync_engine.scheduler.pending) + self.async_engine.active_requests
            middlewares.append(make_load_shed_middleware(_get_load, self._max_pending))

        app = web.Application(middlewares=middlewares)
        app.router.add_route("OPTIONS", "/v1/completions", self._handle_options)
        app.router.add_route("OPTIONS", "/v1/chat/completions", self._handle_options)
        app.router.add_post("/v1/completions", self.handle_completions)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/v1/models/{model_id}", self.handle_model_info)
        app.router.add_post("/v1/tokenize", self.handle_tokenize)
        app.router.add_post("/v1/embeddings", self.handle_embeddings)
        app.router.add_get("/v1/usage", self.handle_usage)
        app.router.add_post("/v1/lora/load", self.handle_lora_load)
        app.router.add_post("/v1/lora/unload", self.handle_lora_unload)
        app.router.add_get("/v1/lora/list", self.handle_lora_list)
        app.router.add_post("/v1/batch", self.handle_batch)
        app.router.add_get("/v1/metrics", self.handle_metrics)
        app.router.add_get("/v1/logs", self.handle_request_log)
        app.router.add_post("/v1/priority", self.handle_priority)
        app.router.add_post("/v1/cancel/{request_id}", self.handle_cancel)
        app.router.add_get("/v1/ws/completions", self.handle_ws_completions)
        app.router.add_get("/docs", self.handle_openapi)
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
        self._shutting_down = True
        await self.async_engine.stop(drain_timeout=30.0)
        logger.info("Server cleanup complete")

    def run(self):
        """Start the server with async continuous batching and graceful shutdown."""
        logger.info(f"vllm-i64 :: {self.model_name}")
        logger.info(f"  http://{self.host}:{self.port}")
        logger.info(f"  POST /v1/completions | POST /v1/chat/completions | GET /health")
        logger.info(f"  POST /v1/batch | GET /v1/models/{{id}} | GET /v1/metrics | GET /v1/logs")
        logger.info(f"  POST /v1/cancel/{{id}} | WS /v1/ws/completions | GET /docs")
        logger.info(f"  mode: async continuous batching")
        app = self.create_app()

        # Graceful shutdown: drain requests before stopping
        async def _graceful_shutdown(app):
            logger.info("Graceful shutdown: stopping new requests...")
            self._shutting_down = True
            # The engine's stop() method handles draining
            logger.info("Graceful shutdown complete")

        app.on_shutdown.append(_graceful_shutdown)

        # Register SIGTERM handler for container/systemd environments
        def _sigterm_handler():
            logger.info("SIGTERM received, initiating graceful shutdown")
            self._shutting_down = True

        try:
            import signal as sig
            loop = asyncio.new_event_loop()
            loop.add_signal_handler(sig.SIGTERM, _sigterm_handler)
        except (NotImplementedError, OSError):
            pass  # Windows doesn't support add_signal_handler

        try:
            web.run_app(app, host=self.host, port=self.port, print=None)
        except KeyboardInterrupt:
            pass  # Clean exit on Ctrl+C — cleanup handlers already ran
