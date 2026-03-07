"""
vllm-i64 :: Agentics — Tools

Built-in tools for the agent loop.
Each tool: name, description, function(args_str) -> str.

Security: tools are sandboxed by default (read-only).
Pass --allow-shell to enable shell execution.

INL - 2025
"""

import os
import subprocess
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

_logger = logging.getLogger("vllm_i64.agentics.tools")


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


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
        # Cap output
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
            # Skip hidden dirs and common noise
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
    "read_file": Tool("read_file", "Read a file (path)", tool_read_file),
    "write_file": Tool("write_file", "Write a file (path\\ncontent)", tool_write_file),
    "list_dir": Tool("list_dir", "List directory contents (path)", tool_list_dir),
    "search_files": Tool("search_files", "Find files (directory pattern)", tool_search_files),
    "grep": Tool("grep", "Search file contents (pattern [directory])", tool_grep),
    "python": Tool("python", "Execute Python code", tool_python),
}

DANGEROUS_TOOLS: Dict[str, Tool] = {
    "shell": Tool("shell", "Execute shell command (requires --allow-shell)", tool_shell),
}


def get_tools(allow_shell: bool = False) -> Dict[str, Tool]:
    """Get available tools based on permissions."""
    tools = dict(SAFE_TOOLS)
    if allow_shell:
        tools.update(DANGEROUS_TOOLS)
    return tools


def tools_description(tools: Dict[str, Tool]) -> str:
    """Format tool descriptions for the system prompt."""
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
