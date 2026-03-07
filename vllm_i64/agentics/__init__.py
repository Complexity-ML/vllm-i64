"""
vllm-i64 :: Agentics

Local AI agent framework powered by vllm-i64 inference.

Features:
  - Native OpenAI tool_calls format
  - Parallel tool execution within a single step
  - Multi-agent orchestrator with task queue
  - ReAct loop with configurable tools

Usage:
    # Single agent:
    agent = Agent(base_url="http://localhost:8000")
    agent.run("your task here")

    # Multi-agent parallel:
    orch = Orchestrator(base_url="http://localhost:8000", max_workers=4)
    orch.submit("task 1")
    orch.submit("task 2")
    results = orch.run_sync()

INL - 2025
"""

from .agent import Agent
from .client import I64Client, ChatMessage
from .tools import Tool, get_tools, execute_tool, execute_tools_parallel, tools_to_openai
from .orchestrator import Orchestrator, AgentTask, TaskResult, TaskStatus

__all__ = [
    "Agent",
    "I64Client", "ChatMessage",
    "Tool", "get_tools", "execute_tool", "execute_tools_parallel", "tools_to_openai",
    "Orchestrator", "AgentTask", "TaskResult", "TaskStatus",
]
