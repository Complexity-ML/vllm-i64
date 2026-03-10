"""
vllm-i64 :: RAG Handlers Mixin

/v1/rag/index, /v1/rag/search, /v1/rag/stats
INL - 2025
"""

import json

from aiohttp import web

from vllm_i64.core.logging import get_logger
from vllm_i64.api.events import AgentEvent

logger = get_logger("vllm_i64.server")


class RAGMixin:

    async def handle_rag_index(self, request: web.Request) -> web.Response:
        """POST /v1/rag/index"""
        if not self.rag_enabled or self.retriever is None:
            return web.json_response({"error": {"message": "RAG not enabled", "type": "server_error"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        text = body.get("text")
        file_path = body.get("file")
        chunk_size = body.get("chunk_size", 200)
        overlap = body.get("overlap", 50)

        if not text and not file_path:
            return web.json_response({"error": {"message": "Provide 'text' or 'file'", "type": "invalid_request_error"}}, status=400)

        session_id = request.headers.get("X-Session-Id", "default")

        try:
            if file_path:
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

            if self._rag_index_path:
                self.retriever.save(self._rag_index_path)

            total = len(self.retriever.vector_index.chunks) if self.retriever.vector_index else n
            self.event_bus.emit(AgentEvent(type="rag_index", session_id=session_id,
                                           data={"chunks_added": n, "total_chunks": total}))
            return web.json_response({"status": "ok", "chunks_added": n, "total_chunks": total})
        except Exception as e:
            logger.error("RAG index error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "RAG indexing failed", "type": "server_error"}}, status=500)

    async def handle_rag_search(self, request: web.Request) -> web.Response:
        """POST /v1/rag/search"""
        if not self.rag_enabled or self.retriever is None:
            return web.json_response({"error": {"message": "RAG not enabled", "type": "server_error"}}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400)

        query = body.get("query")
        k = body.get("k", 3)
        if not query:
            return web.json_response({"error": {"message": "Missing 'query'", "type": "invalid_request_error"}}, status=400)

        session_id = request.headers.get("X-Session-Id", "default")
        try:
            self.event_bus.emit(AgentEvent(type="rag_search", session_id=session_id,
                                           data={"status": "running", "query": query, "k": k}))
            results = self.retriever.retrieve(query, k=k)
            self.event_bus.emit(AgentEvent(type="rag_search", session_id=session_id,
                                           data={"status": "done", "query": query, "count": len(results)}))
            return web.json_response({"query": query, "results": results, "count": len(results)})
        except Exception as e:
            logger.error("RAG search error: %s", e, exc_info=True)
            return web.json_response({"error": {"message": "RAG search failed", "type": "server_error"}}, status=500)

    async def handle_rag_stats(self, request: web.Request) -> web.Response:
        """GET /v1/rag/stats"""
        if not self.rag_enabled or self.retriever is None:
            return web.json_response({"enabled": False})
        idx = self.retriever.vector_index
        if idx is None:
            return web.json_response({"enabled": True, "total_chunks": 0, "dimension": 0,
                                      "index_path": getattr(self, '_rag_index_path', None)})
        return web.json_response({"enabled": True, "total_chunks": len(idx.chunks), "dimension": idx.dim,
                                  "index_path": getattr(self, '_rag_index_path', None)})
