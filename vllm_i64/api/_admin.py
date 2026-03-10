"""
vllm-i64 :: Admin Handlers Mixin

Health, models, tokenize, embeddings, usage, lora, batch,
model-info, metrics, logs, priority, cancel, WebSocket, OpenAPI,
cache stats/purge, monitor, expert stats.
INL - 2025
"""

import asyncio
import json
import time
from typing import List, Optional

from aiohttp import web

from vllm_i64.core.logging import get_logger
from vllm_i64.api.types import CompletionRequest

logger = get_logger("vllm_i64.server")


class AdminMixin:

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health"""
        uptime_s = int(time.monotonic() - self._start_time)
        if self.async_engine is None:
            return web.json_response({"status": "ok", "mode": "sandbox-only",
                                      "uptime_seconds": uptime_s,
                                      "sandbox_enabled": self.sandbox_enabled,
                                      "rag_enabled": self.rag_enabled})
        stats = self.async_engine.get_stats()
        status = "ok"
        checks = {"model_loaded": self.sync_engine.model is not None}
        if self.sync_engine.kv_cache is not None:
            kv = self.sync_engine.kv_cache.get_stats()
            usage = kv["used_blocks"] / max(kv["num_blocks"], 1)
            checks["kv_cache_usage_pct"] = round(usage * 100, 1)
            if usage > 0.95:
                status = "degraded"
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                checks["gpu"] = {"free_mb": round(mem[0]/1e6), "total_mb": round(mem[1]/1e6),
                                 "used_mb": round((mem[1]-mem[0])/1e6),
                                 "utilization_pct": round((1-mem[0]/mem[1])*100,1)}
                if mem[0]/mem[1] < 0.05:
                    status = "degraded"
            else:
                checks["gpu"] = "not_available"
        except Exception as e:
            logger.warning("GPU health check failed: %s", e)
            checks["gpu"] = "error"

        health = {"status": status, "model": self.model_name, "uptime_seconds": uptime_s,
                  "requests_served": self.request_counter, "engine": stats,
                  "queue": {"pending": len(self.sync_engine.scheduler.pending),
                            "running": len(self.sync_engine.scheduler.running),
                            "active_requests": self.async_engine.active_requests},
                  "checks": checks, "cache": self._request_cache.hit_rate_info,
                  "usage": self._usage_tracker.get_total(),
                  "latency": self._latency_tracker.percentiles()}
        if self.sync_engine.kv_cache is not None:
            health["kv_cache"] = self.sync_engine.kv_cache.get_stats()
        return web.json_response(health)

    async def handle_models(self, request: web.Request) -> web.Response:
        """GET /v1/models"""
        return web.json_response({"object": "list", "data": [{"id": self.model_name, "object": "model", "owned_by": "inl"}]})

    async def handle_tokenize(self, request: web.Request) -> web.Response:
        """POST /v1/tokenize"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)
        text = body.get("text")
        messages = body.get("messages")
        if messages:
            text = self._apply_chat_template(messages)
        elif not text:
            return web.json_response({"error": {"message": "Missing 'text' or 'messages'", "type": "invalid_request_error"}}, status=400)
        tokens = self._tokenize(text)
        return web.json_response({"tokens": tokens, "count": len(tokens)})

    async def handle_embeddings(self, request: web.Request) -> web.Response:
        """POST /v1/embeddings"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)
        input_data = body.get("input")
        if not input_data:
            return web.json_response({"error": {"message": "Missing 'input'", "type": "invalid_request_error"}}, status=400)
        if isinstance(input_data, str):
            input_data = [input_data]
        try:
            embeddings = []
            total_tokens = 0
            for i, text in enumerate(input_data):
                token_ids = self._tokenize(text)
                total_tokens += len(token_ids)
                embeddings.append({"object": "embedding", "index": i, "embedding": self.sync_engine.embed(token_ids)})
            return web.json_response({"object": "list", "data": embeddings, "model": self.model_name,
                                      "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}})
        except Exception as e:
            logger.error("Embedding error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "Internal server error", "type": "server_error"}}, status=500)

    async def handle_usage(self, request: web.Request) -> web.Response:
        """GET /v1/usage"""
        auth = request.headers.get("Authorization", "")
        api_key = auth[7:] if auth.startswith("Bearer ") else None
        usage = self._usage_tracker.get(api_key) if api_key else self._usage_tracker.get_total()
        return web.json_response({"usage": usage})

    async def handle_lora_load(self, request: web.Request) -> web.Response:
        """POST /v1/lora/load"""
        denied = self._require_admin(request)
        if denied:
            return denied
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)
        adapter_id = body.get("adapter_id")
        adapter_path = body.get("path")
        if adapter_id is None or adapter_path is None:
            return web.json_response({"error": {"message": "Missing 'adapter_id' or 'path'", "type": "invalid_request_error"}}, status=400)
        import os
        if not os.path.exists(os.path.realpath(adapter_path)):
            return web.json_response({"error": {"message": "Adapter path does not exist", "type": "invalid_request_error"}}, status=400)
        if self.sync_engine._lora_manager is None:
            try:
                self.sync_engine.enable_lora()
            except Exception as e:
                return web.json_response({"error": {"message": f"Failed to enable LoRA: {e}", "type": "server_error"}}, status=500)
        adapter_name = body.get("name", f"adapter-{adapter_id}")
        scaling = body.get("scaling", 1.0)
        if self.sync_engine.load_lora_adapter(int(adapter_id), adapter_name, adapter_path, scaling):
            return web.json_response({"status": "ok", "adapter_id": adapter_id, "name": adapter_name})
        return web.json_response({"error": {"message": "Failed to load adapter", "type": "server_error"}}, status=500)

    async def handle_lora_unload(self, request: web.Request) -> web.Response:
        """POST /v1/lora/unload"""
        denied = self._require_admin(request)
        if denied:
            return denied
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)
        adapter_id = body.get("adapter_id")
        if adapter_id is None:
            return web.json_response({"error": {"message": "Missing 'adapter_id'", "type": "invalid_request_error"}}, status=400)
        self.sync_engine.unload_lora_adapter(int(adapter_id))
        return web.json_response({"status": "ok", "adapter_id": adapter_id})

    async def handle_lora_list(self, request: web.Request) -> web.Response:
        """GET /v1/lora/list"""
        adapters = self.sync_engine.list_lora_adapters()
        return web.json_response({"adapters": [{"id": aid, "name": name} for aid, name in adapters.items()]})

    async def handle_batch(self, request: web.Request) -> web.Response:
        """POST /v1/batch"""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)
        requests_data = body.get("requests")
        if not requests_data or not isinstance(requests_data, list):
            return web.json_response({"error": {"message": "Missing 'requests' array", "type": "invalid_request_error"}}, status=400)
        if len(requests_data) > 128:
            return web.json_response({"error": {"message": "Max 128 requests per batch", "type": "invalid_request_error"}}, status=400)
        auth = request.headers.get("Authorization", "")
        req_api_key = auth[7:] if auth.startswith("Bearer ") else None
        completion_reqs = []
        for i, rd in enumerate(requests_data):
            prompt = rd.get("prompt")
            if not prompt:
                return web.json_response({"error": {"message": f"Request {i}: missing 'prompt'", "type": "invalid_request_error"}}, status=400)
            req = CompletionRequest(
                prompt=prompt, max_tokens=rd.get("max_tokens", 256),
                temperature=rd.get("temperature", 0.8), top_k=rd.get("top_k", 50),
                top_p=rd.get("top_p", 0.9), min_p=rd.get("min_p", 0.0),
                typical_p=rd.get("typical_p", 1.0), repetition_penalty=rd.get("repetition_penalty", 1.1),
                min_tokens=rd.get("min_tokens", 0), seed=rd.get("seed"),
                logit_bias=rd.get("logit_bias"), frequency_penalty=rd.get("frequency_penalty", 0.0),
                presence_penalty=rd.get("presence_penalty", 0.0),
                suppress_first_tokens=self._space_suppress_ids,
            )
            error = req.validate(max_seq_len=self.sync_engine.scheduler.max_seq_len)
            if error:
                return web.json_response({"error": {"message": f"Request {i}: {error}", "type": "invalid_request_error"}}, status=400)
            completion_reqs.append(req)
        results = await asyncio.gather(
            *[self._async_complete(r, api_key=req_api_key, endpoint="/v1/batch") for r in completion_reqs],
            return_exceptions=True,
        )
        responses = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error("Batch request %d error: %s", i, r, exc_info=True)
                responses.append({"index": i, "error": "Internal server error"})
            else:
                responses.append({"index": i, "result": r.to_dict()})
        return web.json_response({"responses": responses})

    async def handle_model_info(self, request: web.Request) -> web.Response:
        """GET /v1/models/{model_id}"""
        model_id = request.match_info.get("model_id", "")
        if model_id != self.model_name:
            return web.json_response({"error": {"message": f"Model '{model_id}' not found", "type": "not_found_error"}}, status=404)
        info = {"id": self.model_name, "object": "model", "owned_by": "inl", "created": int(self._start_time)}
        if self.sync_engine.model is not None and hasattr(self.sync_engine.model, 'config'):
            cfg = self.sync_engine.model.config
            info["config"] = {k: getattr(cfg, k, None) for k in (
                "num_experts", "vocab_size", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim",
            )}
            info["parameters"] = sum(p.numel() for p in self.sync_engine.model.parameters())
            info["dtype"] = str(next(self.sync_engine.model.parameters()).dtype)
        info["engine"] = {
            "max_batch_size": self.sync_engine.scheduler.max_batch_size,
            "max_seq_len": self.sync_engine.scheduler.max_seq_len,
            "kv_cache": self.sync_engine.kv_cache is not None,
            "speculative": self.sync_engine.speculative_decoder is not None,
            "lora": self.sync_engine._lora_manager is not None,
        }
        return web.json_response(info)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """GET /v1/metrics"""
        return web.json_response({
            "latency": self._latency_tracker.get_all_endpoints(),
            "usage": self._usage_tracker.get_total(),
            "cache": self._request_cache.hit_rate_info,
            "uptime_seconds": int(time.monotonic() - self._start_time),
            "requests_served": self.request_counter,
            "engine": self.async_engine.get_stats(),
        })

    async def handle_request_log(self, request: web.Request) -> web.Response:
        """GET /v1/logs"""
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
        """POST /v1/priority"""
        denied = self._require_admin(request)
        if denied:
            return denied
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)
        api_key = body.get("api_key")
        priority = body.get("priority", 0)
        if not api_key:
            return web.json_response({"error": {"message": "Missing 'api_key'", "type": "invalid_request_error"}}, status=400)
        self._priority_manager.set_priority(api_key, int(priority))
        return web.json_response({"status": "ok", "api_key": api_key, "priority": priority})

    async def handle_cancel(self, request: web.Request) -> web.Response:
        """POST /v1/cancel/{request_id}"""
        try:
            request_id = int(request.match_info["request_id"])
        except (KeyError, ValueError):
            return web.json_response({"error": {"message": "Invalid request_id", "type": "invalid_request_error"}}, status=400)
        await self.async_engine.cancel_request(request_id)
        return web.json_response({"status": "ok", "cancelled": request_id})

    async def handle_ws_completions(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket /v1/ws/completions"""
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
                    prompt=prompt, max_tokens=body.get("max_tokens", 256),
                    temperature=body.get("temperature", 0.8), top_k=body.get("top_k", 50),
                    top_p=body.get("top_p", 0.9), stream=True,
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
                    output_ids: List[int] = []
                    prev_text = ""
                    last_token_id = None
                    async for token_id in self.async_engine.generate_stream(
                        prompt_token_ids=prompt_ids, max_new_tokens=req.max_tokens,
                        sampling_params=req.to_sampling_params(tokenizer=self.tokenizer),
                    ):
                        last_token_id = token_id
                        output_ids.append(token_id)
                        full = self._detokenize(output_ids)
                        token_text = full[len(prev_text):]
                        prev_text = full
                        if token_text:
                            await ws.send_json({"id": stream_id, "object": "text_completion.chunk",
                                               "created": created, "model": self.model_name,
                                               "choices": [{"index": 0, "text": token_text, "finish_reason": None}]})
                    eos_id = getattr(self.tokenizer, 'eos_token_id', None)
                    ws_finish = "stop" if last_token_id is None or (eos_id and last_token_id == eos_id) else "length"
                    await ws.send_json({"id": stream_id, "object": "text_completion.chunk",
                                       "created": created, "model": self.model_name,
                                       "choices": [{"index": 0, "text": "", "finish_reason": ws_finish}], "done": True})
                except Exception as e:
                    logger.error("WebSocket error: %s", e, exc_info=True)
                    await ws.send_json({"error": {"message": "Internal server error", "type": "server_error"}})
            elif msg.type == web.WSMsgType.ERROR:
                break
        return ws

    async def handle_openapi(self, request: web.Request) -> web.Response:
        """GET /docs"""
        return web.json_response({
            "openapi": "3.0.3",
            "info": {"title": "vllm-i64 API", "version": "0.1.0",
                     "description": "Integer-first inference engine for token-routed language models"},
            "paths": {
                "/v1/completions": {"post": {"summary": "Create completion"}},
                "/v1/chat/completions": {"post": {"summary": "Create chat completion"}},
                "/v1/batch": {"post": {"summary": "Batch completions"}},
                "/v1/cancel/{request_id}": {"post": {"summary": "Cancel a running request"}},
                "/v1/models": {"get": {"summary": "List models"}},
                "/v1/ws/completions": {"get": {"summary": "WebSocket streaming completions"}},
                "/health": {"get": {"summary": "Health check"}},
                "/v1/metrics": {"get": {"summary": "Latency and usage metrics"}},
            },
        })

    async def handle_cache_stats(self, request: web.Request) -> web.Response:
        """GET /v1/cache/stats"""
        if self.sync_engine.kv_cache is None:
            return web.json_response({"error": "No KV cache initialized"}, status=404)
        stats = self.sync_engine.kv_cache.get_stats()
        stats["usage_pct"] = round(stats["used_blocks"] / max(stats["num_blocks"], 1) * 100, 1)
        return web.json_response(stats)

    async def handle_cache_purge(self, request: web.Request) -> web.Response:
        """POST /v1/cache/purge"""
        denied = self._require_admin(request)
        if denied:
            return denied
        kv = self.sync_engine.kv_cache
        if kv is None:
            return web.json_response({"error": "No KV cache initialized"}, status=404)
        if not kv.prefix_cache_enabled:
            return web.json_response({"error": "Prefix caching not enabled"}, status=400)
        before = len(kv.pool._hash_to_block)
        kv.pool._hash_to_block.clear()
        # Reset block hashes
        for blk in kv.pool.blocks:
            blk.reset_hash()
        return web.json_response({"status": "ok", "purged_blocks": before})

    async def handle_monitor(self, request: web.Request) -> web.Response:
        """GET /v1/monitor"""
        engine = self.sync_engine
        snapshot = {
            "timestamp": time.time(), "uptime_s": int(time.monotonic() - self._start_time),
            "requests_served": self.request_counter,
            "active_requests": self.async_engine.active_requests,
            "peak_batch_size": self.async_engine.peak_batch_size,
            "scheduler": engine.scheduler.get_stats(),
            "engine": {"total_steps": engine.total_steps, "total_tokens_generated": engine.total_tokens_generated},
        }
        if engine.kv_cache is not None:
            kv = engine.kv_cache.get_stats()
            kv["usage_pct"] = round(kv["used_blocks"] / max(kv["num_blocks"], 1) * 100, 1)
            snapshot["kv_cache"] = kv
        if engine.total_steps > 0 and engine._perf_total_ms > 0:
            snapshot["perf"] = {
                "avg_step_ms": round(engine._perf_total_ms / engine.total_steps, 2),
                "tok_per_s": round(engine.total_tokens_generated / (engine._perf_total_ms / 1000), 1),
                "forward_pct": round(engine._perf_forward_ms / engine._perf_total_ms * 100, 1),
            }
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                snapshot["gpu"] = {"free_mb": round(mem[0]/1e6), "total_mb": round(mem[1]/1e6),
                                   "utilization_pct": round((1-mem[0]/mem[1])*100,1)}
        except Exception:
            pass
        if engine._lora_manager is not None:
            adapters = engine.list_lora_adapters()
            snapshot["lora"] = {"loaded_adapters": len(adapters), "adapters": list(adapters.values())}
        return web.json_response(snapshot)

    async def handle_expert_stats(self, request: web.Request) -> web.Response:
        """GET /v1/experts"""
        engine = self.sync_engine
        num_experts = getattr(engine, "num_experts", 0)
        if num_experts <= 1:
            return web.json_response({"error": "Not a MoE model (num_experts <= 1)"}, status=400)
        expert_counts = [0] * num_experts
        total_tokens = 0
        for req in list(engine.scheduler.running) + list(engine.scheduler.finished):
            for tid in req.output_token_ids:
                expert_counts[int(tid) % num_experts] += 1
                total_tokens += 1
        if total_tokens > 0:
            distribution = [round(c / total_tokens, 4) for c in expert_counts]
            response = {"num_experts": num_experts, "total_tokens": total_tokens,
                        "distribution": distribution, "counts": expert_counts,
                        "imbalance": round(max(distribution) - min(distribution), 4)}
            self._last_expert_response = response
            return web.json_response(response)
        if self._last_expert_response is not None:
            return web.json_response(self._last_expert_response)
        return web.json_response({"num_experts": num_experts, "total_tokens": 0,
                                  "distribution": [0.0]*num_experts, "counts": expert_counts})
