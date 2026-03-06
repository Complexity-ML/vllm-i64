"""
vllm-i64 :: Agentics — ReAct Agent

ReAct loop: Thought → Action → Observation → repeat.
Connects to a local vllm-i64 server via OpenAI-compatible API.

Usage:
    agent = Agent(base_url="http://localhost:8000")
    agent.run("Create a Python script that ...")

INL - 2025
"""

import re
import json
import logging
from typing import Dict, List, Optional

from .client import I64Client
from .tools import Tool, get_tools, tools_description, execute_tool

_logger = logging.getLogger("vllm_i64.agentics.agent")

SYSTEM_PROMPT = """\
You are an autonomous AI agent powered by pacific-prime. You solve tasks step by step.

You have access to these tools:
{tools}

To use a tool, output EXACTLY this format (one tool per step):
```tool
{{"tool": "tool_name", "args": "arguments"}}
```

After a tool executes, you'll see its output as an Observation.
Then continue reasoning with another Thought, or give your final Answer.

Format:
Thought: <your reasoning>
Action:
```tool
{{"tool": "...", "args": "..."}}
```

When done, output:
Answer: <your final response to the user>

Rules:
- One tool call per step
- Always think before acting
- If a tool fails, try a different approach
- Give a clear Answer when the task is complete"""

# Regex to extract tool calls
_TOOL_PATTERN = re.compile(
    r"```tool\s*\n?\s*(\{.*?\})\s*\n?\s*```",
    re.DOTALL,
)


class Agent:
    """ReAct agent that uses vllm-i64 as LLM backend."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        allow_shell: bool = False,
        max_steps: int = 20,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        verbose: bool = True,
    ):
        self.client = I64Client(base_url=base_url, api_key=api_key)
        self.tools = get_tools(allow_shell=allow_shell)
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

    def _system_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=tools_description(self.tools)),
        }

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

    def _parse_tool_call(self, text: str) -> Optional[tuple]:
        """Extract tool name and args from assistant response."""
        match = _TOOL_PATTERN.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
            name = data.get("tool", "")
            args = data.get("args", "")
            if isinstance(args, list):
                args = " ".join(str(a) for a in args)
            return name, str(args)
        except (json.JSONDecodeError, KeyError):
            return None

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from assistant response."""
        # Look for "Answer:" at start of line
        match = re.search(r"^Answer:\s*(.+)", text, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def run(self, task: str) -> str:
        """
        Run the agent on a task. Returns the final answer.

        The agent loops: send messages → parse response → execute tool → observe.
        Stops when it outputs an Answer or hits max_steps.
        """
        self._print("[Agent]", f"Task: {task}", "cyan")

        self.history = [
            self._system_message(),
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            self._print(f"\n[Step {step}/{self.max_steps}]", "", "dim")

            # Call LLM
            try:
                response = self.client.chat(
                    messages=self.history,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except ConnectionError as e:
                self._print("[Error]", str(e), "red")
                return f"Error: {e}"

            self._print("[Assistant]", response, "green")

            # Check for final answer
            answer = self._extract_answer(response)
            if answer:
                self._print("\n[Answer]", answer, "cyan")
                self.history.append({"role": "assistant", "content": response})
                return answer

            # Check for tool call
            tool_call = self._parse_tool_call(response)
            if tool_call:
                name, args = tool_call
                self._print(f"[Tool: {name}]", args, "yellow")

                # Execute tool
                result = execute_tool(self.tools, name, args)
                self._print("[Observation]", result[:500], "dim")

                # Add to history
                self.history.append({"role": "assistant", "content": response})
                self.history.append({
                    "role": "user",
                    "content": f"Observation: {result}",
                })
            else:
                # No tool call and no answer — treat as answer
                self.history.append({"role": "assistant", "content": response})
                self._print("\n[Done]", "(no explicit Answer, using response)", "dim")
                return response

        self._print("[Agent]", "Max steps reached", "red")
        return "Error: agent reached maximum steps without completing the task."

    def chat(self, message: str) -> str:
        """Single-turn chat (no tool use, just LLM response)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant powered by pacific-prime."},
            {"role": "user", "content": message},
        ]
        return self.client.chat(messages, temperature=self.temperature, max_tokens=self.max_tokens)

    def interactive(self):
        """Interactive REPL mode."""
        self._print("[Agent]", "Interactive mode. Type /help for commands.", "cyan")
        self.history = [self._system_message()]

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Slash commands
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
                self.history = [self._system_message()]
                print("History cleared.")
                continue
            elif user_input.startswith("/task "):
                task = user_input[6:]
                self.run(task)
                continue
            elif user_input == "/exit":
                break

            # Normal message — add to history and get response
            self.history.append({"role": "user", "content": user_input})

            try:
                response = self.client.chat(
                    messages=self.history,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except ConnectionError as e:
                self._print("[Error]", str(e), "red")
                continue

            # Check for tool call in interactive mode
            tool_call = self._parse_tool_call(response)
            if tool_call:
                name, args = tool_call
                self._print(f"[Tool: {name}]", args, "yellow")
                result = execute_tool(self.tools, name, args)
                self._print("[Observation]", result[:500], "dim")

                self.history.append({"role": "assistant", "content": response})
                self.history.append({"role": "user", "content": f"Observation: {result}"})

                # Get follow-up response
                try:
                    follow_up = self.client.chat(
                        messages=self.history,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    self._print("[Assistant]", follow_up, "green")
                    self.history.append({"role": "assistant", "content": follow_up})
                except ConnectionError:
                    pass
            else:
                self._print("[Assistant]", response, "green")
                self.history.append({"role": "assistant", "content": response})
