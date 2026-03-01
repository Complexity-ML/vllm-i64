"""
vllm-i64 :: Tool/Function Call Parser

Parses function calls from model-generated text.
Supports OpenAI-compatible tool_calls format.

Extraction strategies:
  1. JSON object with "name" and "arguments" fields
  2. <tool_call> XML-style tags (common in fine-tuned models)

INL - 2025
"""

import json
import re
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A parsed function/tool call from model output."""
    id: str
    type: str = "function"
    function_name: str = ""
    function_arguments: str = ""  # JSON string


class ToolCallParser:
    """
    Parse tool/function calls from generated text.

    Tries multiple extraction patterns:
      1. JSON: {"name": "func", "arguments": {...}}
      2. XML tags: <tool_call>{"name": "func", ...}</tool_call>
      3. Function call syntax: func_name(arg1, arg2)
    """

    def __init__(self, tools: List[Dict]):
        self.tools = tools
        self.function_names = set()
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                self.function_names.add(t["function"]["name"])

    def parse(self, text: str) -> Optional[List[ToolCall]]:
        """
        Try to extract function calls from generated text.

        Returns list of ToolCall objects, or None if no calls found.
        """
        calls = []

        # Strategy 1: <tool_call>...</tool_call> tags
        tag_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
        for match in tag_pattern.finditer(text):
            call = self._parse_json_call(match.group(1))
            if call:
                calls.append(call)

        if calls:
            return calls

        # Strategy 2: JSON objects with "name" and "arguments"
        json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}', re.DOTALL)
        for match in json_pattern.finditer(text):
            call = self._parse_json_call(match.group(0))
            if call:
                calls.append(call)

        return calls if calls else None

    def _parse_json_call(self, text: str) -> Optional[ToolCall]:
        """Try to parse a single JSON function call."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        name = data.get("name", "")
        if name not in self.function_names:
            return None

        arguments = data.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function_name=name,
            function_arguments=arguments,
        )
