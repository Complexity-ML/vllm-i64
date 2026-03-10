"""
vllm-i64 :: Agent Handlers Mixin

/v1/execute (sandbox), /v1/agent/events, /v1/agent/history
INL - 2025
"""

import asyncio
import json

from aiohttp import web

from vllm_i64.core.logging import get_logger
from vllm_i64.api.events import AgentEvent

logger = get_logger("vllm_i64.server")


class AgentMixin:

    async def handle_execute(self, request: web.Request) -> web.Response:
        """POST /v1/execute — sandboxed code execution."""
        if not self.sandbox_enabled or self.sandbox is None:
            return web.json_response(
                {"error": {"message": "Sandbox not enabled. Start server with --sandbox", "type": "server_error"}},
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        code = body.get("code")
        if not code or not isinstance(code, str):
            return web.json_response({"error": {"message": "'code' field is required", "type": "invalid_request_error"}}, status=400)

        language = body.get("language", "python")
        if language not in self.sandbox.supported_languages:
            return web.json_response(
                {"error": {"message": f"Unsupported language: {language}. Available: {', '.join(self.sandbox.supported_languages)}",
                           "type": "invalid_request_error"}},
                status=400,
            )

        session_id = request.headers.get("X-Session-Id", "default")
        self.event_bus.emit(AgentEvent(type="sandbox", session_id=session_id,
                                       data={"status": "running", "language": language, "code": code}))

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.sandbox.execute, code, language)

        self.event_bus.emit(AgentEvent(type="sandbox", session_id=session_id,
                                       data={"status": "done", "language": language, **result.to_dict()}))
        return web.json_response(result.to_dict())

    async def handle_agent_events(self, request: web.Request) -> web.StreamResponse:
        """GET /v1/agent/events — SSE stream of agent activity."""
        session_filter = request.query.get("session_id")
        history_count = int(request.query.get("history", "20"))

        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        response.headers.update({"Cache-Control": "no-cache", "Connection": "keep-alive",
                                   "Access-Control-Allow-Origin": "*"})
        await response.prepare(request)

        for event in self.event_bus.get_history(session_id=session_filter, limit=history_count):
            await response.write(f"data: {json.dumps(event)}\n\n".encode())

        sub_id, queue = self.event_bus.subscribe(session_filter=session_filter)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if event is None:
                    break
                if session_filter and event.session_id != session_filter:
                    continue
                await response.write(f"data: {json.dumps(event.to_dict())}\n\n".encode())
        except (ConnectionResetError, ConnectionError):
            pass
        finally:
            self.event_bus.unsubscribe(sub_id)
        return response

    async def handle_agent_history(self, request: web.Request) -> web.Response:
        """GET /v1/agent/history — recent events (JSON)."""
        session_id = request.query.get("session_id")
        limit = int(request.query.get("limit", "50"))
        events = self.event_bus.get_history(session_id=session_id, limit=limit)
        return web.json_response({"events": events, "count": len(events),
                                   "subscribers": self.event_bus.subscriber_count})
