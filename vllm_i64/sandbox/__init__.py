"""
vllm-i64 :: Sandbox

Isolated code execution with resource limits.
Designed to be used as an agent tool or via the /v1/execute API.

Security model:
  - subprocess with resource limits (CPU time, memory, file size)
  - Temporary working directory per execution (deleted after)
  - Network access disabled via environment
  - Configurable timeout (default: 30s)

Usage:
    sandbox = Sandbox(timeout=30, max_memory_mb=256)
    result = sandbox.execute("print(2 + 2)", language="python")
    # result.stdout == "4\n", result.exit_code == 0

INL - 2025
"""

from .executor import Sandbox, ExecutionResult

__all__ = ["Sandbox", "ExecutionResult"]
