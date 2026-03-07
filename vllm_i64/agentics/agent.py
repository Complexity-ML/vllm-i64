"""
vllm-i64 :: Agentics — ReAct Agent

ReAct loop with native OpenAI tool_calls format.
Supports parallel tool execution within a single step.

Modes:
  - agent.run("task")          — autonomous task completion
  - agent.chat("message")     — single-turn chat
  - agent.interactive()       — REPL with tool use
  - await agent.arun("task")  — async version of run()

INL - 2025
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any

from .client import I64Client, ChatMessage
from .tools import (
    Tool, get_tools, tools_to_openai,
    execute_tool_call, execute_tools_parallel,
)

_logger = logging.getLogger("vllm_i64.agentics.agent")

SYSTEM_PROMPT = """\
You are an autonomous AI agent powered by pacific-prime. You solve tasks step by step.

You have access to tools. When you need information or want to take action, call one or more tools.
You can call multiple tools in parallel when they are independent.

When the task is complete, respond with your final answer as plain text (no tool calls).

Rules:
- Think before acting
- Call multiple independent tools at once for efficiency
- If a tool fails, try a different approach
- Give a clear final answer when the task is complete"""


class Agent:
    """
    ReAct agent with native OpenAI tool_calls and parallel execution.

    Uses the vllm-i64 server's tool_calls support for structured output,
    and executes independent tool calls concurrently.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        allow_shell: bool = False,
        max_steps: int = 20,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        verbose: bool = True,
        tools: Optional[Dict[str, Tool]] = None,
    ):
        self.client = I64Client(base_url=base_url, api_key=api_key)
        self.tools = tools or get_tools(allow_shell=allow_shell)
        self.openai_tools = tools_to_openai(self.tools)
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []

    def _print(self, prefix: str, text: str, color: str = ""):
        if not self.verbose:
            return
        colors = {
            "green": "\033[32m",
            "yellow": "\033[33m",
            "cyan": "\033[36m",
            "red": "\033[31m",
            "dim": "\033[2m",
        }
        reset = "\033[0m"
        c = colors.get(color, "")
        print(f"{c}{prefix}{reset} {text}")

    def _call_llm(self) -> ChatMessage:
        """Send current history to the LLM with tools."""
        return self.client.chat(
            messages=self.history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.openai_tools,
            tool_choice="auto",
        )

    # -----------------------------------------------------------------
    # Synchronous run (uses asyncio.run internally for parallel tools)
    # -----------------------------------------------------------------

    def run(self, task: str) -> str:
        """
        Run the agent on a task. Returns the final answer.

        Tool calls within a step are executed in parallel.
        """
        self._print("[Agent]", f"Task: {task}", "cyan")

        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            self._print(f"\n[Step {step}/{self.max_steps}]", "", "dim")

            try:
                response = self._call_llm()
            except (ConnectionError, OSError, TimeoutError) as e:
                self._print("[Error]", str(e), "red")
                return f"Error: {e}"

            # No tool calls → final answer
            if not response.has_tool_calls:
                self._print("[Answer]", response.content or "", "green")
                self.history.append({"role": "assistant", "content": response.content})
                return response.content or ""

            # Tool calls — execute in parallel
            tool_calls = response.tool_calls
            n = len(tool_calls)
            names = [tc["function"]["name"] for tc in tool_calls]
            self._print(f"[Tools x{n}]", ", ".join(names), "yellow")

            # Add assistant message with tool_calls to history
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = tool_calls
            self.history.append(assistant_msg)

            # Execute tools in parallel
            if n == 1:
                # Single tool — no need for asyncio
                result = execute_tool_call(self.tools, tool_calls[0])
                tool_results = [{
                    "role": "tool",
                    "tool_call_id": tool_calls[0]["id"],
                    "content": result,
                }]
            else:
                # Multiple tools — parallel execution
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Already in async context — run sequentially
                    tool_results = []
                    for tc in tool_calls:
                        result = execute_tool_call(self.tools, tc)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })
                else:
                    tool_results = asyncio.run(
                        execute_tools_parallel(self.tools, tool_calls)
                    )

            # Log results and add to history
            for tr in tool_results:
                self._print(f"  [{tr['tool_call_id']}]", tr["content"][:300], "dim")
                self.history.append(tr)

        self._print("[Agent]", "Max steps reached", "red")
        return "Error: agent reached maximum steps without completing the task."

    # -----------------------------------------------------------------
    # Async run — native async for use in async contexts
    # -----------------------------------------------------------------

    async def arun(self, task: str) -> str:
        """
        Async version of run(). Executes tool calls in parallel natively.
        """
        self._print("[Agent]", f"Task: {task}", "cyan")

        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            self._print(f"\n[Step {step}/{self.max_steps}]", "", "dim")

            try:
                # Run sync HTTP call in executor to not block event loop
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, self._call_llm)
            except (ConnectionError, OSError, TimeoutError) as e:
                self._print("[Error]", str(e), "red")
                return f"Error: {e}"

            if not response.has_tool_calls:
                self._print("[Answer]", response.content or "", "green")
                self.history.append({"role": "assistant", "content": response.content})
                return response.content or ""

            tool_calls = response.tool_calls
            names = [tc["function"]["name"] for tc in tool_calls]
            self._print(f"[Tools x{len(tool_calls)}]", ", ".join(names), "yellow")

            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = tool_calls
            self.history.append(assistant_msg)

            # Parallel tool execution
            tool_results = await execute_tools_parallel(self.tools, tool_calls)

            for tr in tool_results:
                self._print(f"  [{tr['tool_call_id']}]", tr["content"][:300], "dim")
                self.history.append(tr)

        self._print("[Agent]", "Max steps reached", "red")
        return "Error: agent reached maximum steps without completing the task."

    # -----------------------------------------------------------------
    # Simple chat (no tools)
    # -----------------------------------------------------------------

    def chat(self, message: str) -> str:
        """Single-turn chat (no tool use, just LLM response)."""
        return self.client.chat_text(
            [
                {"role": "system", "content": "You are a helpful assistant powered by pacific-prime."},
                {"role": "user", "content": message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    # -----------------------------------------------------------------
    # Interactive REPL
    # -----------------------------------------------------------------

    def interactive(self):
        """Interactive REPL mode with tool use."""
        self._print("[Agent]", "Interactive mode. Type /help for commands.", "cyan")
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input == "/help":
                print("Commands:")
                print("  /help     Show this help")
                print("  /tools    List available tools")
                print("  /clear    Clear conversation history")
                print("  /task     Switch to task mode (agent loop)")
                print("  /exit     Exit")
                continue
            elif user_input == "/tools":
                for name, tool in self.tools.items():
                    print(f"  {name}: {tool.description}")
                continue
            elif user_input == "/clear":
                self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("History cleared.")
                continue
            elif user_input.startswith("/task "):
                task = user_input[6:]
                self.run(task)
                continue
            elif user_input == "/exit":
                break

            self.history.append({"role": "user", "content": user_input})

            try:
                response = self._call_llm()
            except (ConnectionError, OSError, TimeoutError) as e:
                self._print("[Error]", str(e), "red")
                continue

            if response.has_tool_calls:
                tool_calls = response.tool_calls
                names = [tc["function"]["name"] for tc in tool_calls]
                self._print(f"[Tools x{len(tool_calls)}]", ", ".join(names), "yellow")

                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response.content or ""}
                assistant_msg["tool_calls"] = tool_calls
                self.history.append(assistant_msg)

                # Execute tools (parallel if multiple)
                if len(tool_calls) == 1:
                    result = execute_tool_call(self.tools, tool_calls[0])
                    tool_results = [{
                        "role": "tool",
                        "tool_call_id": tool_calls[0]["id"],
                        "content": result,
                    }]
                else:
                    try:
                        tool_results = asyncio.run(
                            execute_tools_parallel(self.tools, tool_calls)
                        )
                    except RuntimeError:
                        tool_results = []
                        for tc in tool_calls:
                            result = execute_tool_call(self.tools, tc)
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            })

                for tr in tool_results:
                    self._print(f"  [{tr['tool_call_id']}]", tr["content"][:300], "dim")
                    self.history.append(tr)

                # Follow-up response after tool results
                try:
                    follow_up = self._call_llm()
                    self._print("[Assistant]", follow_up.content or "", "green")
                    self.history.append({"role": "assistant", "content": follow_up.content})
                except (ConnectionError, OSError, TimeoutError):
                    pass
            else:
                self._print("[Assistant]", response.content or "", "green")
                self.history.append({"role": "assistant", "content": response.content})
