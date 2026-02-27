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
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass, asdict

from aiohttp import web

from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine, GenerationResult


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stream: bool = False


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
    ):
        # Wrap sync engine in async engine for continuous batching
        self.async_engine = AsyncI64Engine.__new__(AsyncI64Engine)
        self.async_engine.engine = engine
        self.async_engine._pending_futures = {}
        self.async_engine._request_times = {}
        self.async_engine._engine_task = None
        self.async_engine._running = False
        self.async_engine.active_requests = 0
        self.async_engine.peak_batch_size = 0

        self.sync_engine = engine
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.model_name = model_name
        self.host = host
        self.port = port
        self.request_counter: int = 0

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
            return tmpl.render(messages=messages, add_generation_prompt=True)
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

        return CompletionResponse(
            id=f"cmpl-{self.request_counter}",
            created=int(time.time()),
            model=self.model_name,
            choices=[{
                "text": output_text,
                "index": 0,
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(result.output_tokens),
                    "total_tokens": len(prompt_ids) + len(result.output_tokens),
                    "engine_steps": result.num_steps,
                    "elapsed_ms": round(result.elapsed_ms, 2),
                },
            }],
        )

    async def _async_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion — batched with other concurrent requests."""
        prompt_ids = self._tokenize(request.prompt)

        result = await self.async_engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
        )

        return self._build_response(result, prompt_ids)

    async def _async_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming completion via async engine."""
        prompt_ids = self._tokenize(request.prompt)

        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
        ):
            token_text = self._detokenize([token_id])
            chunk = {
                "id": f"cmpl-{self.request_counter}",
                "object": "text_completion.chunk",
                "choices": [{"text": token_text, "index": 0}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    # =====================================================================
    # aiohttp handlers
    # =====================================================================

    async def handle_completions(self, request: web.Request) -> web.Response:
        """POST /v1/completions — async continuous batching."""
        body = await request.json()
        req = CompletionRequest(
            prompt=body.get("prompt", ""),
            max_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 50),
            top_p=body.get("top_p", 1.0),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            stream=body.get("stream", False),
        )

        if req.stream:
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            await response.prepare(request)
            async for chunk in self._async_stream(req):
                await response.write(chunk.encode())
            return response

        result = await self._async_complete(req)
        return web.json_response(result.to_dict())

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """POST /v1/chat/completions — async continuous batching."""
        body = await request.json()
        messages = body.get("messages", [])
        prompt = self._apply_chat_template(messages)

        req = CompletionRequest(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 50),
            top_p=body.get("top_p", 1.0),
            stream=body.get("stream", False),
        )

        if req.stream:
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            await response.prepare(request)
            async for chunk in self._async_stream(req):
                await response.write(chunk.encode())
            return response

        result = await self._async_complete(req)
        result_dict = result.to_dict()
        if result_dict["choices"]:
            text = result_dict["choices"][0]["text"]
            result_dict["choices"][0] = {
                "message": {"role": "assistant", "content": text},
                "index": 0,
                "finish_reason": "length",
            }
        result_dict["object"] = "chat.completion"
        return web.json_response(result_dict)

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health — includes async engine stats."""
        stats = self.async_engine.get_stats()
        return web.json_response({
            "status": "ok",
            "model": self.model_name,
            "engine": stats,
            "requests_served": self.request_counter,
        })

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

    def create_app(self) -> web.Application:
        """Create aiohttp application with routes and engine lifecycle."""
        app = web.Application()
        app.router.add_post("/v1/completions", self.handle_completions)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)

        # Start/stop async engine with the app
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app):
        """Start the async engine loop when server starts."""
        await self.async_engine.start()
        print(f"  engine: continuous batching started")

    async def _on_cleanup(self, app):
        """Stop the async engine loop when server shuts down."""
        await self.async_engine.stop()

    def run(self):
        """Start the server with async continuous batching."""
        print(f"vllm-i64 :: {self.model_name}")
        print(f"  http://{self.host}:{self.port}")
        print(f"  POST /v1/completions")
        print(f"  POST /v1/chat/completions")
        print(f"  GET  /health")
        print(f"  mode: async continuous batching")
        app = self.create_app()
        web.run_app(app, host=self.host, port=self.port, print=None)
