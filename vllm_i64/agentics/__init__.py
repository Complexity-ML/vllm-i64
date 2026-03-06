"""
vllm-i64 :: Agentics

Local AI agent framework powered by vllm-i64 inference.
ReAct loop with tool use, runs against a local CUDA server.

Usage:
    # Start server first:
    #   vllm-i64 serve pacific-prime-chat --checkpoint ... --port 8000
    #
    # Then run agent:
    #   vllm-i64 agent "your task here"
    #   vllm-i64 agent --interactive

INL - 2025
"""

from .agent import Agent
from .client import I64Client
from .tools import Tool, get_tools, execute_tool

__all__ = ["Agent", "I64Client", "Tool", "get_tools", "execute_tool"]
