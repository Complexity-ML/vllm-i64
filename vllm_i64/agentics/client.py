"""
vllm-i64 :: Agentics — LLM Client

OpenAI-compatible HTTP client for vllm-i64 server.
Supports both streaming and non-streaming chat completions.

INL - 2025
"""

import json
import logging
from typing import List, Dict, Optional, Iterator
from urllib.request import Request, urlopen
from urllib.error import URLError

_logger = logging.getLogger("vllm_i64.agentics.client")


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

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Send chat completion request, return assistant message text."""
        body = {
            "model": "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        data = json.dumps(body).encode()
        req = Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen(req, timeout=120) as resp:
                if stream:
                    return self._read_stream(resp)
                else:
                    result = json.loads(resp.read())
                    return result["choices"][0]["message"]["content"]
        except URLError as e:
            _logger.error("Failed to connect to %s: %s", self.base_url, e)
            raise ConnectionError(f"Cannot reach vllm-i64 server at {self.base_url}") from e

    def _read_stream(self, resp) -> str:
        """Read SSE stream, accumulate and return full text."""
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
        return "".join(chunks)

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

    def health(self) -> bool:
        """Check if server is reachable."""
        try:
            req = Request(f"{self.base_url}/health")
            with urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
