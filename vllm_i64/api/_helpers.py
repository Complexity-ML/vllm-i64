"""
vllm-i64 :: Server Helpers Mixin

Tokenize/detokenize, chat template, image processing, response building.
INL - 2025
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional

from aiohttp import web

from vllm_i64.core.logging import get_logger
from vllm_i64.api.types import CompletionResponse
from vllm_i64.engine.i64_engine import GenerationResult

logger = get_logger("vllm_i64.server")


class HelpersMixin:
    """Shared helpers: tokenization, chat template, image pre-processing, response building."""

    # ------------------------------------------------------------------
    # Request ID
    # ------------------------------------------------------------------

    def _next_request_id(self) -> str:
        n = next(self._request_counter)
        self.request_counter = n
        return f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return [int(b) for b in text.encode("utf-8")]

    async def _tokenize_async(self, text: str) -> List[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenize_pool, self._tokenize, text)

    def _detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        safe_ids = [t & 0xFF for t in token_ids]
        return bytes(safe_ids).decode("utf-8", errors="replace")

    async def _detokenize_async(self, token_ids: List[int]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._tokenize_pool, self._detokenize, token_ids)

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content_text(content) -> str:
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
                        import base64, io
                        from PIL import Image
                        _, b64_data = url.split(",", 1)
                        image = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                        images.append(image)
                    else:
                        logger.warning("Non-base64 image URLs not supported: %s...", url[:60])
                except Exception as e:
                    logger.error("Failed to decode image: %s", e)
        return images

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        normalized = [
            {"role": m.get("role", "user"), "content": self._extract_content_text(m.get("content", ""))}
            for m in messages
        ]

        if self.chat_template:
            from jinja2 import Template
            prompt = Template(self.chat_template).render(messages=normalized, add_generation_prompt=True)
            logger.info("[CHAT] Jinja template applied")
            return prompt

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=True,
                )
                logger.info("[CHAT] HF tokenizer template applied")
                return prompt
            except Exception as e:
                logger.debug("[CHAT] HF apply_chat_template failed: %s", e)

        # Pre-train model — raw concat
        return "".join(m.get("content", "") for m in normalized)

    @staticmethod
    def _chat_stop_sequences(user_stop: Optional[List[str]] = None) -> List[str]:
        return list(user_stop) if user_stop else []

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess_images(self, images: list):
        import torch
        model = getattr(self.sync_engine, 'model', None)
        if model is not None and hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
            return torch.cat([model.vision_encoder.preprocess_image(img) for img in images], dim=0)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return torch.cat([transform(img).unsqueeze(0) for img in images], dim=0)

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    def _build_response(self, result: GenerationResult, prompt_ids: List[int]) -> CompletionResponse:
        output_text = self._detokenize(result.output_tokens)
        choice = {"text": output_text, "index": 0, "finish_reason": result.finish_reason}
        if result.token_logprobs:
            choice["logprobs"] = {
                "tokens": [self._detokenize([lp.token_id]) for lp in result.token_logprobs],
                "token_logprobs": [lp.logprob for lp in result.token_logprobs],
                "top_logprobs": [lp.top_logprobs for lp in result.token_logprobs],
            }
        resp = CompletionResponse(
            id=self._next_request_id(),
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
        )
        resp._usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(result.output_tokens),
            "total_tokens": len(prompt_ids) + len(result.output_tokens),
        }
        resp._engine_metrics = {
            "engine_steps": result.num_steps,
            "elapsed_ms": round(result.elapsed_ms, 2),
        }
        return resp

    # ------------------------------------------------------------------
    # Admin auth
    # ------------------------------------------------------------------

    def _require_admin(self, request: web.Request) -> Optional[web.Response]:
        if not self.api_key:
            return None
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
        if token != self.api_key:
            return web.json_response(
                {"error": {"message": "Admin endpoint requires valid API key", "type": "auth_error"}},
                status=403,
            )
        return None
