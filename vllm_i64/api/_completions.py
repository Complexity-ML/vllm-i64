"""
vllm-i64 :: Completions Mixin

/v1/completions and /v1/chat/completions handlers + async streaming.
INL - 2025
"""

import asyncio
import hashlib
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

from aiohttp import web

from vllm_i64.core.logging import get_logger
from vllm_i64.api.types import CompletionRequest

logger = get_logger("vllm_i64.server")


class CompletionsMixin:

    # ------------------------------------------------------------------
    # Core async generation
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_namespace(api_key: Optional[str]) -> Optional[bytes]:
        """Derive a 16-byte KV cache namespace from the API key.

        Blocks are hashed per-namespace so different API keys never share
        cached KV blocks — eliminates cross-user timing oracle attacks.
        Returns None when no API key is present (anonymous requests share
        an open namespace among themselves, which is acceptable).
        """
        if not api_key:
            return None
        return hashlib.sha256(api_key.encode()).digest()[:16]

    async def _async_complete(
        self,
        request: CompletionRequest,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/completions",
    ):
        t0 = time.monotonic()
        prompt_ids = await self._tokenize_async(request.prompt)
        pixel_values = getattr(request, '_pixel_values', None)
        ns = self._cache_namespace(api_key)

        result = await self.async_engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            pixel_values=pixel_values,
            cache_namespace=ns,
        )

        if len(result.output_tokens) <= 3:
            eos_cfg = getattr(self.async_engine.engine.model, 'config', None)
            eos_id = getattr(eos_cfg, 'eos_token_id', '?') if eos_cfg else '?'
            logger.warning(
                "[DEBUG] Short generation: prompt_ids=%s output_tokens=%s eos_config=%s finish=%s",
                prompt_ids[-5:], result.output_tokens, eos_id, result.finish_reason,
            )

        resp = self._build_response(result, prompt_ids)
        latency_ms = (time.monotonic() - t0) * 1000
        from vllm_i64.api.types import compute_partition
        partition = compute_partition(api_key, getattr(request, "user", None))
        self._usage_tracker.record(api_key or "", len(prompt_ids), len(result.output_tokens))
        self._latency_tracker.record(endpoint, latency_ms)
        self._request_logger.log_request(
            endpoint=endpoint, status=200, latency_ms=latency_ms,
            prompt_tokens=len(prompt_ids), completion_tokens=len(result.output_tokens),
            api_key=api_key, request_id=resp.id, partition=partition,
        )
        return resp

    async def _async_stream(self, request: CompletionRequest, api_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        prompt_ids = await self._tokenize_async(request.prompt)
        stream_id = self._next_request_id()
        created = int(time.time())
        output_ids: List[int] = []
        prev_text = ""
        last_token_id = None
        ns = self._cache_namespace(api_key)
        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            cache_namespace=ns,
        ):
            last_token_id = token_id
            output_ids.append(token_id)
            full_text = self._detokenize(output_ids)
            token_text = full_text[len(prev_text):]
            prev_text = full_text
            if not token_text:
                continue
            yield f"data: {json.dumps({'id': stream_id, 'object': 'text_completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'text': token_text, 'finish_reason': None}]})}\n\n"

        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        finish_reason = "stop" if last_token_id is None or (eos_id is not None and last_token_id == eos_id) else "length"
        yield f"data: {json.dumps({'id': stream_id, 'object': 'text_completion', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': finish_reason}]})}\n\n"
        yield "data: [DONE]\n\n"

    async def _async_chat_stream(
        self, request: CompletionRequest, tools: Optional[list] = None, api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        prompt_ids = await self._tokenize_async(request.prompt)
        stream_id = self._next_request_id()
        created = int(time.time())
        ns = self._cache_namespace(api_key)

        yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

        output_ids: List[int] = []
        prev_text = ""
        last_token_id = None
        pixel_values = getattr(request, '_pixel_values', None)
        async for token_id in self.async_engine.generate_stream(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            sampling_params=request.to_sampling_params(tokenizer=self.tokenizer),
            pixel_values=pixel_values,
            cache_namespace=ns,
        ):
            last_token_id = token_id
            output_ids.append(token_id)
            full_text = self._detokenize(output_ids)
            token_text = full_text[len(prev_text):]
            prev_text = full_text
            if not token_text:
                continue
            yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}]})}\n\n"

        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        finish_reason = "stop" if last_token_id is None or (eos_id is not None and last_token_id == eos_id) else "length"
        yield f"data: {json.dumps({'id': stream_id, 'object': 'chat.completion.chunk', 'created': created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def handle_completions(self, request: web.Request) -> web.Response:
        """POST /v1/completions"""
        if self.async_engine is None:
            return web.json_response({"error": {"message": "No model loaded", "type": "server_error"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        prompt = body.get("prompt")
        if not prompt:
            return web.json_response({"error": {"message": "Missing 'prompt'", "type": "invalid_request_error"}}, status=400)

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
        error = req.validate(max_seq_len=self.sync_engine.scheduler.max_seq_len)
        if error:
            return web.json_response({"error": {"message": error, "type": "invalid_request_error"}}, status=400)

        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None

        try:
            if req.stream:
                response = web.StreamResponse()
                response.content_type = "text/event-stream"
                response.headers["Cache-Control"] = "no-cache"
                await response.prepare(request)
                gen = self._async_stream(req, api_key=req_api_key)
                try:
                    async for chunk in gen:
                        await response.write(chunk.encode())
                except (ConnectionResetError, ConnectionError):
                    await gen.aclose()
                return response

            cache_kwargs = dict(
                temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                min_p=req.min_p, typical_p=req.typical_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                seed=req.seed,
            )
            cached = self._request_cache.get(req.prompt, req.max_tokens, **cache_kwargs)
            if cached is not None:
                return web.json_response(cached)

            result = await self._async_complete(req, api_key=req_api_key)
            result_dict = result.to_dict()
            self._request_cache.put(req.prompt, req.max_tokens, result_dict, **cache_kwargs)
            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            return web.Response(status=499, text="Client disconnected")
        except Exception as e:
            logger.error("Completion error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "Internal server error", "type": "server_error"}}, status=500)

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """POST /v1/chat/completions"""
        if self.async_engine is None:
            return web.json_response({"error": {"message": "No model loaded", "type": "server_error"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        messages = body.get("messages")
        if not messages:
            return web.json_response({"error": {"message": "Missing 'messages'", "type": "invalid_request_error"}}, status=400)

        prompt = self._apply_chat_template(messages)

        images = self._extract_images_from_messages(messages)
        pixel_values = None
        if images:
            pixel_values = self._preprocess_images(images)

        if body.get("rag") and self.rag_enabled and self.retriever is not None:
            user_query = messages[-1].get("content", "")
            if isinstance(user_query, list):
                user_query = self._extract_content_text(user_query)
            if user_query:
                context = self.retriever.get_context(user_query, k=body.get("rag_k", 3))
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
        req._pixel_values = pixel_values

        error = req.validate(max_seq_len=self.sync_engine.scheduler.max_seq_len)
        if error:
            return web.json_response({"error": {"message": error, "type": "invalid_request_error"}}, status=400)

        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None

        try:
            if req.stream:
                response = web.StreamResponse()
                response.content_type = "text/event-stream"
                response.headers["Cache-Control"] = "no-cache"
                await response.prepare(request)
                gen = self._async_chat_stream(req, body.get("tools"), api_key=req_api_key)
                try:
                    async for chunk in gen:
                        await response.write(chunk.encode())
                        await response.drain()
                except (ConnectionResetError, ConnectionError):
                    await gen.aclose()
                return response

            result = await self._async_complete(req, api_key=req_api_key, endpoint="/v1/chat/completions")
            result_dict = result.to_dict()
            if result_dict["choices"]:
                text = result_dict["choices"][0]["text"]
                finish_reason = result_dict["choices"][0].get("finish_reason", "length")
                message = {"role": "assistant", "content": text}
                tools = body.get("tools")
                if tools:
                    from vllm_i64.core.tool_parser import ToolCallParser
                    tool_calls = ToolCallParser(tools).parse(text)
                    if tool_calls:
                        message["tool_calls"] = [
                            {"id": tc.id, "type": tc.type, "function": {"name": tc.function_name, "arguments": tc.function_arguments}}
                            for tc in tool_calls
                        ]
                        finish_reason = "tool_calls"
                chat_choice = {"message": message, "index": 0, "finish_reason": finish_reason}
                if "logprobs" in result_dict["choices"][0]:
                    chat_choice["logprobs"] = result_dict["choices"][0]["logprobs"]
                result_dict["choices"][0] = chat_choice
            result_dict["object"] = "chat.completion"
            return web.json_response(result_dict)
        except (ConnectionResetError, ConnectionError):
            return web.Response(status=499, text="Client disconnected")
        except Exception as e:
            logger.error("Chat completion error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "Internal server error", "type": "server_error"}}, status=500)
