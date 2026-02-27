"""
vllm-i64 :: API Server

OpenAI-compatible API for token-routed inference.
Requests come in as text → tokenized to i64 → engine runs → i64 tokens → text out.

INL - 2025
"""

import json
import time
import asyncio
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass, asdict

from vllm_i64.engine.i64_engine import I64Engine, GenerationResult


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
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
    Inference server wrapping the I64Engine.

    The server handles:
    - Tokenization (text → i64 token IDs)
    - Engine dispatch (i64 batch scheduling)
    - Detokenization (i64 token IDs → text)
    - Streaming responses

    All internal operations are integer.
    String handling is only at the API boundary.
    """

    def __init__(
        self,
        engine: I64Engine,
        tokenizer=None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.request_counter: int = 0  # Integer counter

    def _tokenize(self, text: str) -> List[int]:
        """Text → i64 token IDs. Boundary operation."""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        # Fallback: use bytes as token IDs (for testing)
        return [int(b) for b in text.encode("utf-8")]

    def _detokenize(self, token_ids: List[int]) -> str:
        """i64 token IDs → text. Boundary operation."""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        # Fallback
        return bytes(token_ids).decode("utf-8", errors="replace")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Synchronous completion.

        Flow:
          text (str) → tokenize → i64 token IDs
          → engine.generate (all i64 internally)
          → i64 token IDs → detokenize → text (str)
        """
        # String → integer boundary
        prompt_ids = self._tokenize(request.prompt)

        # Pure integer zone
        result = self.engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
        )

        # Integer → string boundary
        output_text = self._detokenize(result.output_tokens)

        self.request_counter += 1

        return CompletionResponse(
            id=f"cmpl-{self.request_counter}",
            created=int(time.time()),
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

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion (SSE).

        Each token is generated and yielded as it comes.
        """
        prompt_ids = self._tokenize(request.prompt)
        request_id = self.engine.add_request(prompt_ids, request.max_tokens)

        token_count = 0
        while True:
            results = self.engine.step()

            if request_id in results:
                token_id = results[request_id]
                token_text = self._detokenize([token_id])
                token_count += 1

                chunk = {
                    "id": f"cmpl-{self.request_counter}",
                    "object": "text_completion.chunk",
                    "choices": [{"text": token_text, "index": 0}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Check if done (integer comparison)
            done = any(
                r.request_id == request_id
                for r in self.engine.scheduler.finished
            )
            if done:
                yield "data: [DONE]\n\n"
                break

            await asyncio.sleep(0)

    def health(self) -> Dict:
        """Health check with integer stats."""
        stats = self.engine.get_stats()
        return {
            "status": "ok",
            "engine": stats,
            "requests_served": self.request_counter,
        }
