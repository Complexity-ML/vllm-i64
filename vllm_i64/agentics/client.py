"""
vllm-i64 :: Agentics — LLM Client

OpenAI-compatible HTTP client for vllm-i64 server.
Supports sync + async, streaming, and native tool_calls.

INL - 2025
"""

import json
import asyncio
import logging
from typing import List, Dict, Optional, Iterator, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

_logger = logging.getLogger("vllm_i64.agentics.client")


class SearchResponse:
    """Structured search completion response."""

    __slots__ = ("content", "sources", "query", "finish_reason")

    def __init__(
        self,
        content: str = "",
        sources: Optional[List[Dict]] = None,
        query: str = "",
        finish_reason: str = "stop",
    ):
        self.content = content
        self.sources = sources or []
        self.query = query
        self.finish_reason = finish_reason

    def __repr__(self):
        return f"SearchResponse(query={self.query!r}, sources={len(self.sources)}, content={self.content[:60]!r}...)"


class ChatMessage:
    """Structured chat completion response."""

    __slots__ = ("role", "content", "tool_calls", "finish_reason")

    def __init__(
        self,
        role: str = "assistant",
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        finish_reason: str = "stop",
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def __repr__(self):
        if self.has_tool_calls:
            names = [tc["function"]["name"] for tc in self.tool_calls]
            return f"ChatMessage(tool_calls={names})"
        return f"ChatMessage(content={self.content[:60]!r}...)" if self.content and len(self.content) > 60 else f"ChatMessage(content={self.content!r})"


class I64Client:
    """HTTP client for a local vllm-i64 server."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, endpoint: str, body: dict, timeout: int = 120) -> dict:
        """POST JSON to the server and return parsed response."""
        data = json.dumps(body).encode()
        req = Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except URLError as e:
            _logger.error("Failed to connect to %s: %s", self.base_url, e)
            raise ConnectionError(f"Cannot reach vllm-i64 server at {self.base_url}") from e

    # -----------------------------------------------------------------
    # Chat completions — full OpenAI-compatible interface
    # -----------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> ChatMessage:
        """
        Send chat completion request, return structured ChatMessage.

        Args:
            messages: OpenAI-format messages (role/content, or role/tool_call_id/content for tool results)
            tools: OpenAI-format tool definitions
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}
            stream: if True, consumes SSE stream and returns accumulated result
        """
        body: Dict[str, Any] = {
            "model": "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        if stream:
            body["stream"] = True

        if stream:
            return self._chat_stream(body)

        result = self._post("/v1/chat/completions", body)
        choice = result["choices"][0]
        msg = choice["message"]

        return ChatMessage(
            role=msg.get("role", "assistant"),
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def chat_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Simple text-only chat (backwards compatible)."""
        msg = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return msg.content or ""

    def _chat_stream(self, body: dict) -> ChatMessage:
        """Send streaming request, accumulate and return full ChatMessage."""
        data = json.dumps(body).encode()
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(req, timeout=120) as resp:
                chunks = []
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks.append(content)
                    except (json.JSONDecodeError, KeyError):
                        continue
                return ChatMessage(content="".join(chunks))
        except URLError as e:
            raise ConnectionError(f"Cannot reach vllm-i64 server at {self.base_url}") from e

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream chat completion, yield text chunks."""
        body = {
            "model": "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        data = json.dumps(body).encode()
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )

        with urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError):
                    continue

    # -----------------------------------------------------------------
    # Search completions — Perplexity-style web search + generation
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        search_count: int = 5,
        user: Optional[str] = None,
        stream: bool = False,
    ) -> SearchResponse:
        """
        Search-augmented generation: query → web search → cited answer.

        Args:
            query: The search query
            search_count: Number of web results to fetch
            user: User ID for team key isolation
            stream: If True, consumes SSE stream
        """
        body: Dict[str, Any] = {
            "query": query,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "search_count": search_count,
            "stream": stream,
        }
        if user:
            body["user"] = user

        if stream:
            return self._search_stream(body, query)

        result = self._post("/v1/search/completions", body)
        choice = result["choices"][0]
        return SearchResponse(
            content=choice["message"]["content"],
            sources=result.get("sources", []),
            query=result.get("query", query),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def _search_stream(self, body: dict, query: str) -> SearchResponse:
        """Send streaming search request, accumulate result + sources."""
        data = json.dumps(body).encode()
        req = Request(
            f"{self.base_url}/v1/search/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(req, timeout=120) as resp:
                chunks = []
                sources = []
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        # Sources event (sent after generation)
                        if "sources" in data:
                            sources = data["sources"]
                            continue
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks.append(content)
                    except (json.JSONDecodeError, KeyError):
                        continue
                return SearchResponse(
                    content="".join(chunks),
                    sources=sources,
                    query=query,
                )
        except URLError as e:
            raise ConnectionError(f"Cannot reach vllm-i64 server at {self.base_url}") from e

    def stream_search(
        self,
        query: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        search_count: int = 5,
        user: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream search completion, yield text chunks (sources in last chunk)."""
        body = {
            "query": query,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "search_count": search_count,
            "stream": True,
        }
        if user:
            body["user"] = user
        data = json.dumps(body).encode()
        req = Request(
            f"{self.base_url}/v1/search/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        with urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    if "sources" in chunk:
                        continue
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError):
                    continue

    # -----------------------------------------------------------------
    # Search history
    # -----------------------------------------------------------------

    def search_history(self, user: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get search history for the authenticated user."""
        params = f"?limit={limit}"
        if user:
            params += f"&user={user}"
        req = Request(
            f"{self.base_url}/v1/search/history{params}",
            headers=self._headers(),
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data.get("history", [])
        except URLError as e:
            _logger.error("Failed to get search history: %s", e)
            return []

    def clear_search_history(self, user: Optional[str] = None) -> int:
        """Clear search history. Returns count of removed entries."""
        params = f"?user={user}" if user else ""
        req = Request(
            f"{self.base_url}/v1/search/history{params}",
            headers=self._headers(),
            method="DELETE",
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data.get("removed", 0)
        except URLError as e:
            _logger.error("Failed to clear search history: %s", e)
            return 0

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    def health(self) -> bool:
        """Check if server is reachable."""
        try:
            req = Request(f"{self.base_url}/health")
            with urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
