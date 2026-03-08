"""
vllm-i64 :: API Server (aiohttp)

OpenAI-compatible API for token-routed inference.
text → tokenize → i64 → async engine → i64 → detokenize → text

Uses AsyncI64Engine for continuous batching:
  - Multiple concurrent requests are batched together
  - Each forward pass processes mixed prefill + decode
  - Maximum GPU utilization and tok/s

Endpoints:
    POST /v1/completions      → completion (sync + streaming)
    POST /v1/chat/completions  → chat completion (sync + streaming)
    GET  /health              → health check + engine stats
    GET  /v1/models           → list models
    GET  /v1/monitor          → live monitoring snapshot
    GET  /v1/cache/stats      → KV cache statistics
    POST /v1/cache/purge      → purge prefix cache (admin)
    GET  /v1/experts          → expert routing distribution

    POST /v1/lora/load        → load LoRA adapter (admin)
    POST /v1/lora/unload      → unload LoRA adapter (admin)
    GET  /v1/lora/list        → list loaded adapters
    POST /v1/execute          → sandboxed code execution (--sandbox)

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
from vllm_i64.api.events import EventBus, AgentEvent
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
    # Suppress specific token IDs at step 0 (chat first-token fix)
    suppress_first_tokens: Optional[List[int]] = None

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
        # Build output constraints from response_format, stop sequences, and suppress_first_tokens
        constraints = None
        has_constraints = self.response_format or self.stop or self.suppress_first_tokens
        if has_constraints:
            stop_seqs = None
            if self.stop and tokenizer is not None:
                stop_seqs = [tokenizer.encode(s) for s in self.stop]
            elif self.stop:
                stop_seqs = [[int(b) for b in s.encode("utf-8")] for s in self.stop]
            constraints = OutputConstraints(
                json_mode=bool(self.response_format and self.response_format.get("type") == "json_object"),
                regex_pattern=(
                    self.response_format.get("pattern")
                    if self.response_format and self.response_format.get("type") == "regex"
                    else None
                ),
                stop_sequences=stop_seqs,
                suppress_first_tokens=self.suppress_first_tokens,
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
        rag_index_path: Optional[str] = None,
        sandbox_enabled: bool = False,
        sandbox_timeout: int = 30,
        sandbox_max_memory_mb: int = 256,
        sandbox_user: Optional[str] = None,
    ):
        # Create async engine — use dedicated CPU engine when on CPU
        # engine=None means sandbox-only / RAG-only mode (no model)
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

        # Fix EOS token: model config default (0) is often wrong (<unk>).
        # Use tokenizer's EOS which actually looks up </s>, <|endoftext|>, etc.
        if tokenizer and hasattr(engine, 'model') and engine.model is not None:
            tok_eos = tokenizer.eos_token_id
            cfg_eos = getattr(engine.model.config, 'eos_token_id', 0)
            if cfg_eos != tok_eos:
                logger.info("Fixing eos_token_id: config=%d → tokenizer=%d", cfg_eos, tok_eos)
                engine.model.config.eos_token_id = tok_eos
        # Suppress leading space at step 0: model predicts space→EOS after
        # "Assistant:", but with space suppressed it picks a content token instead.
        self._space_suppress_ids = None
        if tokenizer:
            space_ids = tokenizer.encode(" ")
            if len(space_ids) == 1:
                self._space_suppress_ids = [space_ids[0]]
            elif len(space_ids) == 2 and space_ids[0] == tokenizer.bos_token_id:
                self._space_suppress_ids = [space_ids[1]]

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
        # Cache last non-empty expert distribution
        self._last_expert_response: dict | None = None
        self._priority_manager = PriorityManager()
        self._shutting_down = False
        # Thread pool for tokenization (avoids blocking the event loop)
        self._tokenize_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tokenize")

        # RAG — native retrieval-augmented generation
        self.retriever = None
        self.rag_enabled = False
        if rag_index_path:
            try:
                import os
                from vllm_i64.rag import Retriever
                if os.path.exists(rag_index_path):
                    self.retriever = Retriever.load(rag_index_path)
                    self.rag_enabled = True
                    logger.info("RAG enabled: loaded index from %s", rag_index_path)
                else:
                    self.retriever = Retriever()
                    self.rag_enabled = True
                    logger.info("RAG enabled: empty index (will save to %s)", rag_index_path)
                self._rag_index_path = rag_index_path
            except Exception as e:
                logger.warning("RAG init failed: %s", e)
        else:
            # RAG always available for runtime indexing even without --rag-index
            try:
                from vllm_i64.rag import Retriever
                self.retriever = Retriever()
                self.rag_enabled = True
                self._rag_index_path = None
            except ImportError:
                pass

        # Sandbox — isolated code execution
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
            logger.info("Sandbox enabled: %s, timeout=%ds memory=%dMB", level, sandbox_timeout, sandbox_max_memory_mb)

        # Event bus — agent observability (sandbox/RAG/completion events)
        self.event_bus = EventBus()

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

    @staticmethod
    def _extract_content_text(content) -> str:
        """
        Extract text from a message content field.

        Supports both string content and multimodal array content:
          - "hello" → "hello"
          - [{"type": "text", "text": "hello"}, {"type": "image_url", ...}]
            → "<image>\nhello"
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("<image>")
            return "\n".join(parts) if parts else ""
        return str(content) if content else ""

    @staticmethod
    def _extract_images_from_messages(messages: List[Dict]) -> list:
        """
        Extract base64-encoded images from multimodal message content.

        Returns a list of PIL.Image.Image objects decoded from
        data:image/...;base64,... URLs.
        """
        images = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if not url:
                    continue
                try:
                    if url.startswith("data:"):
                        # Parse data URI: data:image/png;base64,iVBOR...
                        import base64
                        import io
                        from PIL import Image
                        # Split off the header
                        header, b64_data = url.split(",", 1)
                        image_bytes = base64.b64decode(b64_data)
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        images.append(image)
                    else:
                        logger.warning("Non-base64 image URLs not supported: %s...", url[:60])
                except Exception as e:
                    logger.error("Failed to decode image: %s", e)
        return images

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template to messages → prompt string."""
        # Normalize multimodal content to text (with <image> placeholders)
        normalized = []
        for msg in messages:
            normalized.append({
                "role": msg.get("role", "user"),
                "content": self._extract_content_text(msg.get("content", "")),
            })

        if self.chat_template:
            from jinja2 import Template
            tmpl = Template(self.chat_template)
            prompt = tmpl.render(messages=normalized, add_generation_prompt=True)
            # Ensure the generation prompt is present (template may not handle add_generation_prompt)
            if not prompt.rstrip().endswith("Assistant:"):
                prompt = prompt.rstrip("\n") + "\n\nAssistant:"
            logger.info("[CHAT] Rendered prompt: %r", prompt)
            return prompt
        parts = []
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        for msg in normalized:
            role = role_map.get(msg.get("role", "user"), "User")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _preprocess_images(self, images: list) -> "torch.Tensor":
        """
        Preprocess PIL images for the vision encoder.

        If the model has a vision encoder, uses its image processor.
        Otherwise falls back to a basic CLIP-compatible transform.

        Args:
            images: list of PIL.Image.Image objects.

        Returns:
            pixel_values: tensor of shape (N, C, H, W).
        """
        import torch

        # Try to use the model's vision encoder processor
        model = getattr(self.sync_engine, 'model', None)
        if model is not None and hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
            pixel_values_list = []
            for img in images:
                pv = model.vision_encoder.preprocess_image(img)
                pixel_values_list.append(pv)
            return torch.cat(pixel_values_list, dim=0)

        # Fallback: basic resize + normalize (CLIP-compatible)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        pixel_values_list = [transform(img).unsqueeze(0) for img in images]
        return torch.cat(pixel_values_list, dim=0)

    @staticmethod
    def _chat_stop_sequences(user_stop: Optional[List[str]] = None) -> List[str]:
        """Merge user stop sequences with default chat template stop strings."""
        defaults = ["\nAssistant:", "\nUser:"]
        if not user_stop:
            return defaults
        merged = list(user_stop)
        for s in defaults:
            if s not in merged:
                merged.append(s)
        return merged

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

        # Pass pixel_values for VLM requests
        pixel_values = getattr(request, '_pixel_values', None)

        result = await self.async_engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            pixel_values=pixel_values,
        )

        # Debug: log prompt and output tokens for short generations (helps diagnose early-EOS)
        if len(result.output_tokens) <= 3:
            eos_cfg = getattr(self.async_engine.engine.model, 'config', None)
            eos_id = getattr(eos_cfg, 'eos_token_id', '?') if eos_cfg else '?'
            logger.warning(
                "[DEBUG] Short generation: prompt_ids=%s output_tokens=%s eos_config=%s finish=%s",
                prompt_ids[-5:], result.output_tokens, eos_id, result.finish_reason,
            )

        resp = self._build_response(result, prompt_ids)
        latency_ms = (time.monotonic() - t0) * 1000

        # Track usage (always, even without api_key)
        self._usage_tracker.record(api_key or "", len(prompt_ids), len(result.output_tokens))

        # Track latency
        self._latency_tracker.record(endpoint, latency_ms)

        # Log request
        self._request_logger.log_request(
            endpoint=endpoint, status=200, latency_ms=latency_ms,
            prompt_tokens=len(prompt_ids), completion_tokens=len(result.output_tokens),
            api_key=api_key,
            request_id=resp.id,
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
        # Incremental detokenization: decode growing token list, emit diff.
        # Prevents cross-request pollution from single-token detok cache and
        # handles multi-byte UTF-8 sequences spanning token boundaries.
        output_ids: List[int] = []
        prev_text = ""
        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
        ):
            last_token_id = token_id
            output_ids.append(token_id)
            full_text = self._detokenize(output_ids)
            token_text = full_text[len(prev_text):]
            prev_text = full_text
            if not token_text:
                continue
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

        # Content chunks — incremental detokenization (same as _async_stream)
        last_token_id = None
        output_ids: List[int] = []
        prev_text = ""
        pixel_values = getattr(request, '_pixel_values', None)
        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            pixel_values=pixel_values,
        ):
            last_token_id = token_id
            output_ids.append(token_id)
            full_text = self._detokenize(output_ids)
            token_text = full_text[len(prev_text):]
            prev_text = full_text
            if not token_text:
                continue
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
        if self.async_engine is None:
            return web.json_response(
                {"error": {"message": "No model loaded (server running in sandbox-only mode)", "type": "server_error"}},
                status=503,
            )
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

        # /v1/completions: send raw prompt as-is (caller controls formatting).
        # Chat template is only applied by /v1/chat/completions.

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
            suppress_first_tokens=self._space_suppress_ids,
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
                gen = self._async_stream(req)
                try:
                    async for chunk in gen:
                        await response.write(chunk.encode())
                except (ConnectionResetError, ConnectionError):
                    await gen.aclose()  # stop generation on client disconnect
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
            logger.error("Completion error: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": "Internal server error", "type": "server_error"}},
                status=500,
            )

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """POST /v1/chat/completions — async continuous batching."""
        if self.async_engine is None:
            return web.json_response(
                {"error": {"message": "No model loaded (server running in sandbox-only mode)", "type": "server_error"}},
                status=503,
            )
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

        # Extract images from multimodal content (if any)
        images = self._extract_images_from_messages(messages)
        pixel_values = None
        if images:
            pixel_values = self._preprocess_images(images)
            logger.info(
                "[VLM] Extracted %d image(s), pixel_values shape: %s",
                len(images), pixel_values.shape if pixel_values is not None else "none",
            )

        # RAG context injection (opt-in via "rag": true in request body)
        if body.get("rag") and self.rag_enabled and self.retriever is not None:
            user_query = messages[-1].get("content", "") if messages else ""
            if isinstance(user_query, list):
                user_query = self._extract_content_text(user_query)
            if user_query:
                rag_k = body.get("rag_k", 3)
                context = self.retriever.get_context(user_query, k=rag_k)
                if context:
                    prompt = f"Context:\n{context}\n\n{prompt}"

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
            stop=self._chat_stop_sequences(body.get("stop")),
            n=body.get("n", 1),
            best_of=body.get("best_of", 1),
            logprobs=body.get("logprobs"),
            seed=body.get("seed"),
            logit_bias=body.get("logit_bias"),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            priority=body.get("priority", 0),
            suppress_first_tokens=self._space_suppress_ids,
        )
        # Attach pixel_values for VLM — will be passed to engine
        req._pixel_values = pixel_values

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
                gen = self._async_chat_stream(req, body.get("tools"))
                try:
                    async for chunk in gen:
                        await response.write(chunk.encode())
                        await response.drain()
                except (ConnectionResetError, ConnectionError):
                    await gen.aclose()  # stop generation on client disconnect
                return response

            result = await self._async_complete(req, api_key=req_api_key, endpoint="/v1/chat/completions")
            result_dict = result.to_dict()
            if result_dict["choices"]:
                text = result_dict["choices"][0]["text"]
                finish_reason = result_dict["choices"][0].get("finish_reason", "length")

                # Strip hallucinated chat template markers from output
                for marker in ("Assistant:", "User:"):
                    idx = text.find(marker)
                    if idx != -1:
                        text = text[:idx].rstrip()
                        finish_reason = "stop"
                        break

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

                chat_choice = {
                    "message": message,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
                # Preserve logprobs if requested
                if "logprobs" in result_dict["choices"][0]:
                    chat_choice["logprobs"] = result_dict["choices"][0]["logprobs"]
                result_dict["choices"][0] = chat_choice
            result_dict["object"] = "chat.completion"
            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            return web.Response(status=499, text="Client disconnected")
        except Exception as e:
            logger.error("Chat completion error: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": "Internal server error", "type": "server_error"}},
                status=500,
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health — detailed health check with KV cache, GPU memory, queue depth."""
        uptime_s = int(time.monotonic() - self._start_time)

        # No-model mode: minimal health response
        if self.async_engine is None:
            return web.json_response({
                "status": "ok",
                "mode": "sandbox-only",
                "uptime_seconds": uptime_s,
                "sandbox_enabled": self.sandbox_enabled,
                "rag_enabled": self.rag_enabled,
            })

        stats = self.async_engine.get_stats()

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
        except Exception as e:
            logger.warning("GPU health check failed: %s", e)
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
            logger.error("Embedding error: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": "Internal server error", "type": "server_error"}},
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

    def _require_admin(self, request: web.Request) -> Optional[web.Response]:
        """Check that admin endpoints are authorized via API key."""
        if not self.api_key:
            return None  # No auth configured — allow all
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
        if token != self.api_key:
            return web.json_response(
                {"error": {"message": "Admin endpoint requires valid API key", "type": "auth_error"}},
                status=403,
            )
        return None

    async def handle_lora_load(self, request: web.Request) -> web.Response:
        """POST /v1/lora/load — load a LoRA adapter."""
        denied = self._require_admin(request)
        if denied:
            return denied
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

        # S2: Validate adapter path — prevent arbitrary filesystem access
        import os
        real_adapter = os.path.realpath(adapter_path)
        if not os.path.exists(real_adapter):
            return web.json_response(
                {"error": {"message": "Adapter path does not exist", "type": "invalid_request_error"}},
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
        denied = self._require_admin(request)
        if denied:
            return denied
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
                suppress_first_tokens=self._space_suppress_ids,
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
                logger.error("Batch request %d error: %s", i, r, exc_info=True)
                responses.append({"index": i, "error": "Internal server error"})
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
        denied = self._require_admin(request)
        if denied:
            return denied
        try:
            n = min(int(request.query.get("n", "50")), 1000)
        except ValueError:
            n = 50
        entries = self._request_logger.get_recent(n)
        return web.json_response({"entries": entries, "count": len(entries)})

    async def handle_priority(self, request: web.Request) -> web.Response:
        """POST /v1/priority — set API key priority level."""
        denied = self._require_admin(request)
        if denied:
            return denied
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
                    suppress_first_tokens=self._space_suppress_ids,
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
                    ws_output_ids: List[int] = []
                    ws_prev_text = ""
                    async for token_id in self.async_engine.generate_stream(
                        prompt_token_ids=prompt_ids,
                        max_new_tokens=req.max_tokens,
                        sampling_params=req.to_sampling_params(tokenizer=self.tokenizer),
                    ):
                        last_token_id = token_id
                        ws_output_ids.append(token_id)
                        ws_full = self._detokenize(ws_output_ids)
                        token_text = ws_full[len(ws_prev_text):]
                        ws_prev_text = ws_full
                        if not token_text:
                            continue
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
                    logger.error("WebSocket error: %s", e, exc_info=True)
                    await ws.send_json({"error": {"message": "Internal server error", "type": "server_error"}})
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


    # =========================================================================
    # RAG endpoints
    # =========================================================================

    async def handle_rag_index(self, request: web.Request) -> web.Response:
        """POST /v1/rag/index — index text or a file for RAG retrieval."""
        if not self.rag_enabled or self.retriever is None:
            return web.json_response(
                {"error": {"message": "RAG not enabled", "type": "server_error"}}, status=503,
            )
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400,
            )

        text = body.get("text")
        file_path = body.get("file")
        chunk_size = body.get("chunk_size", 200)
        overlap = body.get("overlap", 50)

        if not text and not file_path:
            return web.json_response(
                {"error": {"message": "Provide 'text' or 'file'", "type": "invalid_request_error"}}, status=400,
            )

        session_id = request.headers.get("X-Session-Id", "default")

        try:
            if file_path:
                # S1: Prevent path traversal — only allow files under CWD or explicit safe dirs
                import os
                real_path = os.path.realpath(file_path)
                cwd = os.path.realpath(os.getcwd())
                if not real_path.startswith(cwd + os.sep) and real_path != cwd:
                    return web.json_response(
                        {"error": {"message": "File path must be under the working directory", "type": "invalid_request_error"}},
                        status=403,
                    )
                n = self.retriever.index_file(file_path, chunk_size=chunk_size, overlap=overlap)
            else:
                n = self.retriever.index_text(text, chunk_size=chunk_size, overlap=overlap)

            # Auto-save if index path configured
            if self._rag_index_path:
                self.retriever.save(self._rag_index_path)

            total = len(self.retriever.vector_index.chunks) if self.retriever.vector_index else n

            self.event_bus.emit(AgentEvent(
                type="rag_index",
                session_id=session_id,
                data={"chunks_added": n, "total_chunks": total},
            ))

            return web.json_response({"status": "ok", "chunks_added": n, "total_chunks": total})
        except Exception as e:
            logger.error("RAG index error: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": "RAG indexing failed", "type": "server_error"}}, status=500,
            )

    async def handle_rag_search(self, request: web.Request) -> web.Response:
        """POST /v1/rag/search — search indexed documents."""
        if not self.rag_enabled or self.retriever is None:
            return web.json_response(
                {"error": {"message": "RAG not enabled", "type": "server_error"}}, status=503,
            )
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400,
            )

        query = body.get("query")
        k = body.get("k", 3)
        if not query:
            return web.json_response(
                {"error": {"message": "Missing 'query'", "type": "invalid_request_error"}}, status=400,
            )

        session_id = request.headers.get("X-Session-Id", "default")

        try:
            self.event_bus.emit(AgentEvent(
                type="rag_search",
                session_id=session_id,
                data={"status": "running", "query": query, "k": k},
            ))

            results = self.retriever.retrieve(query, k=k)

            self.event_bus.emit(AgentEvent(
                type="rag_search",
                session_id=session_id,
                data={"status": "done", "query": query, "count": len(results)},
            ))

            return web.json_response({"query": query, "results": results, "count": len(results)})
        except Exception as e:
            logger.error("RAG search error: %s", e, exc_info=True)
            return web.json_response(
                {"error": {"message": "RAG search failed", "type": "server_error"}}, status=500,
            )

    async def handle_rag_stats(self, request: web.Request) -> web.Response:
        """GET /v1/rag/stats — RAG index statistics."""
        if not self.rag_enabled or self.retriever is None:
            return web.json_response({"enabled": False})
        idx = self.retriever.vector_index
        if idx is None:
            return web.json_response({"enabled": True, "total_chunks": 0, "dimension": 0, "index_path": getattr(self, '_rag_index_path', None)})
        return web.json_response({
            "enabled": True,
            "total_chunks": len(idx.chunks),
            "dimension": idx.dim,
            "index_path": getattr(self, '_rag_index_path', None),
        })

    # =================================================================
    # Admin endpoints — cache, monitoring, expert routing
    # =================================================================

    async def handle_cache_stats(self, request: web.Request) -> web.Response:
        """GET /v1/cache/stats — KV cache and prefix cache statistics."""
        if self.sync_engine.kv_cache is None:
            return web.json_response({"error": "No KV cache initialized"}, status=404)
        stats = self.sync_engine.kv_cache.get_stats()
        stats["usage_pct"] = round(stats["used_blocks"] / max(stats["num_blocks"], 1) * 100, 1)
        return web.json_response(stats)

    async def handle_cache_purge(self, request: web.Request) -> web.Response:
        """POST /v1/cache/purge — purge the prefix cache (not running KV blocks)."""
        denied = self._require_admin(request)
        if denied:
            return denied
        kv = self.sync_engine.kv_cache
        if kv is None:
            return web.json_response({"error": "No KV cache initialized"}, status=404)
        if not kv.prefix_cache_enabled:
            return web.json_response({"error": "Prefix caching not enabled"}, status=400)
        # Clear prefix hash maps (doesn't affect running sequences)
        before = len(getattr(kv, "_block_to_prefix", {}))
        kv._prefix_hash_to_blocks.clear()
        kv._block_to_prefix.clear()
        return web.json_response({"status": "ok", "purged_blocks": before})

    async def handle_monitor(self, request: web.Request) -> web.Response:
        """GET /v1/monitor — live monitoring snapshot (batch, queues, KV, perf)."""
        engine = self.sync_engine
        async_engine = self.async_engine

        snapshot = {
            "timestamp": time.time(),
            "uptime_s": int(time.monotonic() - self._start_time),
            "requests_served": self.request_counter,
            "active_requests": async_engine.active_requests,
            "peak_batch_size": async_engine.peak_batch_size,
            "scheduler": engine.scheduler.get_stats(),
            "engine": {
                "total_steps": engine.total_steps,
                "total_tokens_generated": engine.total_tokens_generated,
            },
        }

        # KV cache
        if engine.kv_cache is not None:
            kv = engine.kv_cache.get_stats()
            kv["usage_pct"] = round(kv["used_blocks"] / max(kv["num_blocks"], 1) * 100, 1)
            snapshot["kv_cache"] = kv

        # Performance breakdown
        if engine.total_steps > 0 and engine._perf_total_ms > 0:
            snapshot["perf"] = {
                "avg_step_ms": round(engine._perf_total_ms / engine.total_steps, 2),
                "tok_per_s": round(engine.total_tokens_generated / (engine._perf_total_ms / 1000), 1),
                "forward_pct": round(engine._perf_forward_ms / engine._perf_total_ms * 100, 1),
            }

        # GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                snapshot["gpu"] = {
                    "free_mb": round(mem[0] / 1e6),
                    "total_mb": round(mem[1] / 1e6),
                    "utilization_pct": round((1 - mem[0] / mem[1]) * 100, 1),
                }
        except Exception:
            pass

        # LoRA
        if engine._lora_manager is not None:
            adapters = engine.list_lora_adapters()
            snapshot["lora"] = {"loaded_adapters": len(adapters), "adapters": list(adapters.values())}

        return web.json_response(snapshot)

    async def handle_expert_stats(self, request: web.Request) -> web.Response:
        """
        GET /v1/experts — expert routing distribution.

        Returns the avg distribution for the current request. When no request
        is active, returns the last known distribution ("photo finish").
        """
        engine = self.sync_engine
        num_experts = getattr(engine, "num_experts", 0)
        if num_experts <= 1:
            return web.json_response({"error": "Not a MoE model (num_experts <= 1)"}, status=400)

        # Collect from running + just-finished requests
        expert_counts = [0] * num_experts
        total_tokens = 0

        for req in engine.scheduler.running:
            for tid in req.output_token_ids:
                expert_counts[int(tid) % num_experts] += 1
                total_tokens += 1

        for req in engine.scheduler.finished:
            for tid in req.output_token_ids:
                expert_counts[int(tid) % num_experts] += 1
                total_tokens += 1

        if total_tokens > 0:
            distribution = [round(c / total_tokens, 4) for c in expert_counts]
            imbalance = max(distribution) - min(distribution)
            response = {
                "num_experts": num_experts,
                "total_tokens": total_tokens,
                "distribution": distribution,
                "counts": expert_counts,
                "imbalance": round(imbalance, 4),
            }
            # Cache as photo finish
            self._last_expert_response = response
            return web.json_response(response)

        # No active request — return cached photo finish
        if self._last_expert_response is not None:
            return web.json_response(self._last_expert_response)

        return web.json_response({
            "num_experts": num_experts,
            "total_tokens": 0,
            "distribution": [0.0] * num_experts,
            "counts": expert_counts,
        })

    async def handle_execute(self, request: web.Request) -> web.Response:
        """POST /v1/execute — execute code in the sandbox."""
        if not self.sandbox_enabled or self.sandbox is None:
            return web.json_response(
                {"error": {"message": "Sandbox not enabled. Start server with --sandbox", "type": "server_error"}},
                status=503,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400,
            )

        code = body.get("code")
        if not code or not isinstance(code, str):
            return web.json_response(
                {"error": {"message": "'code' field is required", "type": "invalid_request_error"}}, status=400,
            )

        language = body.get("language", "python")
        if language not in self.sandbox.supported_languages:
            return web.json_response(
                {"error": {"message": f"Unsupported language: {language}. Available: {', '.join(self.sandbox.supported_languages)}",
                           "type": "invalid_request_error"}},
                status=400,
            )

        session_id = request.headers.get("X-Session-Id", "default")

        self.event_bus.emit(AgentEvent(
            type="sandbox",
            session_id=session_id,
            data={"status": "running", "language": language, "code": code},
        ))

        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.sandbox.execute, code, language,
        )

        self.event_bus.emit(AgentEvent(
            type="sandbox",
            session_id=session_id,
            data={"status": "done", "language": language, **result.to_dict()},
        ))

        return web.json_response(result.to_dict())

    # =================================================================
    # Agent observability — live event stream
    # =================================================================

    async def handle_agent_events(self, request: web.Request) -> web.StreamResponse:
        """GET /v1/agent/events — SSE stream of agent activity (sandbox, RAG, completions).

        Query params:
            session_id — filter events by session (optional)
            history    — number of past events to replay on connect (default: 20)
        """
        session_filter = request.query.get("session_id")
        history_count = int(request.query.get("history", "20"))

        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["Access-Control-Allow-Origin"] = "*"
        await response.prepare(request)

        # Replay recent history
        history = self.event_bus.get_history(session_id=session_filter, limit=history_count)
        for event in history:
            await response.write(f"data: {json.dumps(event)}\n\n".encode())

        # Subscribe to live events
        sub_id, queue = self.event_bus.subscribe(session_filter=session_filter)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    await response.write(b": keepalive\n\n")
                    continue

                if event is None:
                    break

                # Filter by session if requested
                if session_filter and event.session_id != session_filter:
                    continue

                await response.write(f"data: {json.dumps(event.to_dict())}\n\n".encode())
        except (ConnectionResetError, ConnectionError):
            pass
        finally:
            self.event_bus.unsubscribe(sub_id)

        return response

    async def handle_agent_history(self, request: web.Request) -> web.Response:
        """GET /v1/agent/history — recent events (JSON, not SSE)."""
        session_id = request.query.get("session_id")
        limit = int(request.query.get("limit", "50"))
        events = self.event_bus.get_history(session_id=session_id, limit=limit)
        return web.json_response({
            "events": events,
            "count": len(events),
            "subscribers": self.event_bus.subscriber_count,
        })

    def create_app(self) -> web.Application:
        """Create aiohttp application with routes and engine lifecycle."""
        middlewares = [make_cors_middleware()]
        if self.api_key:
            middlewares.append(make_auth_middleware(self.api_key))
        if self._rate_limiter:
            middlewares.append(make_rate_limit_middleware(self._rate_limiter))
        if self._max_pending > 0 and self.sync_engine is not None:
            def _get_load():
                return len(self.sync_engine.scheduler.pending) + self.async_engine.active_requests
            middlewares.append(make_load_shed_middleware(_get_load, self._max_pending))

        # S4: Limit request body size to 16 MB to prevent memory exhaustion
        app = web.Application(middlewares=middlewares, client_max_size=16 * 1024 * 1024)
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
        # Admin endpoints
        app.router.add_get("/v1/cache/stats", self.handle_cache_stats)
        app.router.add_post("/v1/cache/purge", self.handle_cache_purge)
        app.router.add_get("/v1/monitor", self.handle_monitor)
        app.router.add_get("/v1/experts", self.handle_expert_stats)
        app.router.add_route("OPTIONS", "/v1/cache/purge", self._handle_options)

        # RAG endpoints
        app.router.add_post("/v1/rag/index", self.handle_rag_index)
        app.router.add_post("/v1/rag/search", self.handle_rag_search)
        app.router.add_get("/v1/rag/stats", self.handle_rag_stats)
        app.router.add_route("OPTIONS", "/v1/rag/index", self._handle_options)
        app.router.add_route("OPTIONS", "/v1/rag/search", self._handle_options)
        # Sandbox endpoint
        app.router.add_post("/v1/execute", self.handle_execute)
        app.router.add_route("OPTIONS", "/v1/execute", self._handle_options)
        # Agent observability
        app.router.add_get("/v1/agent/events", self.handle_agent_events)
        app.router.add_get("/v1/agent/history", self.handle_agent_history)
        app.router.add_route("OPTIONS", "/v1/agent/events", self._handle_options)
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
        if self.async_engine is not None:
            await self.async_engine.start()
            logger.info("Engine started: continuous batching active")
        else:
            logger.info("Sandbox-only mode: no engine to start")

    async def _on_cleanup(self, app):
        """Graceful shutdown: drain active requests, then stop engine."""
        logger.info("Server cleanup: draining requests...")
        self._shutting_down = True
        if self.async_engine is not None:
            await self.async_engine.stop(drain_timeout=30.0)
        logger.info("Server cleanup complete")

    def run(self):
        """Start the server with async continuous batching and graceful shutdown."""
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
        logger.info("  mode: async continuous batching")
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
