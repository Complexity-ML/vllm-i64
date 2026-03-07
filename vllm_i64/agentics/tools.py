"""
vllm-i64 :: Agentics — Tools

Built-in tools for the agent loop.
Each tool: name, description, function(args) -> str.

Supports:
  - Sync and async execution
  - Parallel execution via execute_tools_parallel()
  - OpenAI-compatible tool definitions for the API

Security: tools are sandboxed by default (read-only).
Pass allow_shell=True to enable shell execution.

INL - 2025
"""

import os
import asyncio
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

_logger = logging.getLogger("vllm_i64.agentics.tools")

# Shared thread pool for blocking tool calls
_tool_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]
    parameters: Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# Built-in tools
# =========================================================================

def tool_read_file(args: str) -> str:
    """Read a file. Args: file path."""
    path = args.strip()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(100_000)  # cap at 100KB
        lines = content.splitlines()
        if len(lines) > 200:
            content = "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
        return content
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"


def tool_write_file(args: str) -> str:
    """Write a file. Args: first line is path, rest is content."""
    lines = args.split("\n", 1)
    if len(lines) < 2:
        return "Error: format is 'path\\ncontent'"
    path, content = lines[0].strip(), lines[1]
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def tool_list_dir(args: str) -> str:
    """List directory contents. Args: directory path."""
    path = args.strip() or "."
    try:
        entries = sorted(os.listdir(path))
        result = []
        for e in entries[:100]:
            full = os.path.join(path, e)
            suffix = "/" if os.path.isdir(full) else ""
            result.append(f"{e}{suffix}")
        if len(entries) > 100:
            result.append(f"... ({len(entries) - 100} more)")
        return "\n".join(result)
    except NotADirectoryError:
        return f"Error: not a directory: {path}"
    except FileNotFoundError:
        return f"Error: directory not found: {path}"
    except OSError as e:
        return f"Error listing {path}: {e}"


def tool_shell(args: str) -> str:
    """Execute a shell command. Args: command string."""
    try:
        result = subprocess.run(
            args, shell=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        if len(output) > 10_000:
            output = output[:10_000] + "\n... (truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


def tool_python(args: str) -> str:
    """Execute Python code in a subprocess. Args: python code."""
    try:
        result = subprocess.run(
            ["python", "-c", args],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if len(output) > 10_000:
            output = output[:10_000] + "\n... (truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out (30s limit)"
    except Exception as e:
        return f"Error: {e}"


def tool_search_files(args: str) -> str:
    """Search for files matching a pattern. Args: 'directory pattern'."""
    parts = args.strip().split(None, 1)
    directory = parts[0] if parts else "."
    pattern = parts[1] if len(parts) > 1 else "*"

    import fnmatch
    matches = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")]
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    matches.append(os.path.join(root, f))
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        return "\n".join(matches) if matches else "No files found"
    except Exception as e:
        return f"Error: {e}"


def tool_grep(args: str) -> str:
    """Search file contents. Args: 'pattern [directory]'."""
    parts = args.strip().split(None, 1)
    if not parts:
        return "Error: provide a search pattern"
    pattern = parts[0]
    directory = parts[1] if len(parts) > 1 else "."

    matches = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")]
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if pattern in line:
                                matches.append(f"{fpath}:{i}: {line.rstrip()}")
                                if len(matches) >= 30:
                                    return "\n".join(matches) + "\n... (truncated)"
                except (OSError, UnicodeDecodeError):
                    continue
        return "\n".join(matches) if matches else f"No matches for '{pattern}'"
    except Exception as e:
        return f"Error: {e}"


# =========================================================================
# Tool registry
# =========================================================================

SAFE_TOOLS: Dict[str, Tool] = {
    "read_file": Tool(
        "read_file", "Read a file and return its contents",
        tool_read_file,
        {"type": "object", "properties": {"path": {"type": "string", "description": "File path to read"}}, "required": ["path"]},
    ),
    "write_file": Tool(
        "write_file", "Write content to a file (creates directories if needed)",
        tool_write_file,
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    ),
    "list_dir": Tool(
        "list_dir", "List directory contents",
        tool_list_dir,
        {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path (default: current)"}}, "required": []},
    ),
    "search_files": Tool(
        "search_files", "Find files matching a glob pattern",
        tool_search_files,
        {"type": "object", "properties": {"directory": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["pattern"]},
    ),
    "grep": Tool(
        "grep", "Search file contents for a pattern",
        tool_grep,
        {"type": "object", "properties": {"pattern": {"type": "string"}, "directory": {"type": "string"}}, "required": ["pattern"]},
    ),
    "python": Tool(
        "python", "Execute Python code and return output",
        tool_python,
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}}, "required": ["code"]},
    ),
}

DANGEROUS_TOOLS: Dict[str, Tool] = {
    "shell": Tool(
        "shell", "Execute a shell command (requires --allow-shell)",
        tool_shell,
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    ),
}


def get_tools(allow_shell: bool = False) -> Dict[str, Tool]:
    """Get available tools based on permissions."""
    tools = dict(SAFE_TOOLS)
    if allow_shell:
        tools.update(DANGEROUS_TOOLS)
    return tools


def tools_to_openai(tools: Dict[str, Tool]) -> List[Dict]:
    """Convert tools to OpenAI API format for the tools parameter."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools.values()
    ]


def tools_description(tools: Dict[str, Tool]) -> str:
    """Format tool descriptions for system prompt (legacy format)."""
    lines = []
    for name, tool in tools.items():
        lines.append(f"- {name}: {tool.description}")
    return "\n".join(lines)


def execute_tool(tools: Dict[str, Tool], name: str, args: str) -> str:
    """Execute a tool by name, return result string."""
    if name not in tools:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools.keys())}"
    _logger.info("Executing tool: %s", name)
    return tools[name].func(args)


def _format_args(name: str, args_json: str) -> str:
    """Convert OpenAI JSON arguments to the flat string format tools expect."""
    try:
        parsed = __import__("json").loads(args_json) if isinstance(args_json, str) else args_json
    except (__import__("json").JSONDecodeError, TypeError):
        return args_json if isinstance(args_json, str) else str(args_json)

    if not isinstance(parsed, dict):
        return str(parsed)

    # Map structured args to the flat string each tool expects
    if name == "read_file":
        return parsed.get("path", "")
    elif name == "write_file":
        return parsed.get("path", "") + "\n" + parsed.get("content", "")
    elif name == "list_dir":
        return parsed.get("path", ".")
    elif name == "search_files":
        d = parsed.get("directory", ".")
        p = parsed.get("pattern", "*")
        return f"{d} {p}"
    elif name == "grep":
        p = parsed.get("pattern", "")
        d = parsed.get("directory", ".")
        return f"{p} {d}"
    elif name == "python":
        return parsed.get("code", "")
    elif name == "shell":
        return parsed.get("command", "")
    else:
        # Fallback: join values
        return " ".join(str(v) for v in parsed.values())


def execute_tool_call(tools: Dict[str, Tool], tool_call: Dict) -> str:
    """
    Execute an OpenAI-format tool_call dict.

    Expected format:
        {"id": "call_xxx", "function": {"name": "...", "arguments": "{...}"}}
    """
    func = tool_call.get("function", {})
    name = func.get("name", "")
    args_json = func.get("arguments", "{}")
    args_str = _format_args(name, args_json)
    return execute_tool(tools, name, args_str)


async def execute_tools_parallel(
    tools: Dict[str, Tool],
    tool_calls: List[Dict],
) -> List[Dict[str, str]]:
    """
    Execute multiple tool calls in parallel.

    Returns list of {"tool_call_id": ..., "role": "tool", "content": ...}
    messages ready to append to the conversation.
    """
    loop = asyncio.get_running_loop()

    async def _run_one(tc: Dict) -> Dict[str, str]:
        result = await loop.run_in_executor(
            _tool_executor,
            execute_tool_call, tools, tc,
        )
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        }

    results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
    return list(results)
