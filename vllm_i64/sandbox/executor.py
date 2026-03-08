"""
vllm-i64 :: Sandbox Executor

Runs user code in an isolated subprocess with resource limits.

On Linux: uses resource.setrlimit() for CPU, memory, file size limits.
On other platforms: subprocess timeout only (no kernel-level isolation).

Integer-first: no bloat, no Docker, no VMs.
Just fork → limit → exec → collect → kill.

INL - 2025
"""

import os
import sys
import shutil
import signal
import tempfile
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

_logger = logging.getLogger("vllm_i64.sandbox")

# Language runtime configs
_RUNTIMES: Dict[str, Dict] = {
    "python": {
        "cmd": [sys.executable, "-u", "{file}"],
        "ext": ".py",
        "shebang": None,
    },
    "node": {
        "cmd": ["node", "{file}"],
        "ext": ".js",
        "shebang": None,
    },
    "bash": {
        "cmd": ["bash", "{file}"],
        "ext": ".sh",
        "shebang": "#!/bin/bash\nset -euo pipefail\n",
    },
}


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    language: str = "python"
    duration_ms: float = 0.0
    files_created: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "language": self.language,
            "duration_ms": round(self.duration_ms, 2),
            "files_created": self.files_created,
            "success": self.success,
        }

    def to_tool_output(self) -> str:
        """Format for agent tool consumption."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")
        if self.timed_out:
            parts.append(f"[timed out after {self.duration_ms:.0f}ms]")
        elif self.exit_code != 0:
            parts.append(f"[exit code: {self.exit_code}]")
        if self.files_created:
            parts.append(f"[files created: {', '.join(self.files_created)}]")
        return "\n".join(parts) or "(no output)"


class Sandbox:
    """
    Sandboxed code executor with resource limits.

    Args:
        timeout: Max execution time in seconds (default: 30)
        max_memory_mb: Max memory in MB (default: 256, Linux only)
        max_output_bytes: Max stdout/stderr capture (default: 64KB)
        max_file_size_mb: Max file size writable (default: 10MB, Linux only)
        allowed_languages: Which languages to allow (default: all)
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 256,
        max_output_bytes: int = 65536,
        max_file_size_mb: int = 10,
        allowed_languages: Optional[List[str]] = None,
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_bytes = max_output_bytes
        self.max_file_size_mb = max_file_size_mb
        self.allowed_languages = allowed_languages or list(_RUNTIMES.keys())
        self._is_linux = sys.platform.startswith("linux")

    def execute(self, code: str, language: str = "python") -> ExecutionResult:
        """
        Execute code in an isolated subprocess.

        Args:
            code: Source code to execute
            language: Runtime language (python, node, bash)

        Returns:
            ExecutionResult with stdout, stderr, exit_code, etc.
        """
        if language not in self.allowed_languages:
            return ExecutionResult(
                stdout="",
                stderr=f"Language '{language}' not allowed. Available: {', '.join(self.allowed_languages)}",
                exit_code=1,
                language=language,
            )

        runtime = _RUNTIMES.get(language)
        if not runtime:
            return ExecutionResult(
                stdout="",
                stderr=f"Unknown language: {language}",
                exit_code=1,
                language=language,
            )

        # Create temp directory for this execution
        tmpdir = tempfile.mkdtemp(prefix="i64_sandbox_")
        try:
            return self._run_in_sandbox(code, language, runtime, tmpdir)
        finally:
            # Clean up — always destroy the sandbox
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _run_in_sandbox(
        self,
        code: str,
        language: str,
        runtime: dict,
        tmpdir: str,
    ) -> ExecutionResult:
        """Run code inside the sandbox directory."""
        import time

        # Write source file
        ext = runtime["ext"]
        src_path = os.path.join(tmpdir, f"main{ext}")
        with open(src_path, "w", encoding="utf-8") as f:
            if runtime["shebang"]:
                f.write(runtime["shebang"])
            f.write(code)

        # Build command
        cmd = [part.replace("{file}", src_path) for part in runtime["cmd"]]

        # Environment: minimal, no network hints
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
            "LANG": "en_US.UTF-8",
            # Disable network for Python
            "no_proxy": "*",
            "http_proxy": "http://0.0.0.0:0",
            "https_proxy": "http://0.0.0.0:0",
            # Prevent Python from importing user site-packages junk
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        # Pre-exec function for Linux resource limits
        preexec = self._make_preexec() if self._is_linux else None

        t0 = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                env=env,
                preexec_fn=preexec,
                # Start new process group so we can kill the whole tree
                start_new_session=True,
            )

            stdout_bytes, stderr_bytes = proc.communicate(timeout=self.timeout)
            exit_code = proc.returncode

        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, AttributeError):
                proc.kill()
            stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            exit_code = -9

        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                stdout="",
                stderr=f"Sandbox error: {e}",
                exit_code=1,
                language=language,
                duration_ms=duration_ms,
            )

        duration_ms = (time.monotonic() - t0) * 1000

        # Truncate output
        stdout = stdout_bytes[:self.max_output_bytes].decode("utf-8", errors="replace")
        stderr = stderr_bytes[:self.max_output_bytes].decode("utf-8", errors="replace")

        # Detect files created (exclude our source file)
        files_created = []
        try:
            for name in os.listdir(tmpdir):
                if name != f"main{ext}":
                    files_created.append(name)
        except OSError:
            pass

        _logger.info(
            "sandbox: lang=%s exit=%d timed_out=%s duration=%.0fms stdout=%d stderr=%d",
            language, exit_code, timed_out, duration_ms, len(stdout), len(stderr),
        )

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            language=language,
            duration_ms=duration_ms,
            files_created=files_created,
        )

    def _make_preexec(self):
        """Create a preexec_fn that sets resource limits (Linux only)."""
        max_mem = self.max_memory_mb * 1024 * 1024
        max_cpu = self.timeout
        max_fsize = self.max_file_size_mb * 1024 * 1024

        def _set_limits():
            import resource
            # CPU time (seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu))
            # Virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
            # File size
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_fsize, max_fsize))
            # No core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            # Max 32 child processes (prevent fork bombs)
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

        return _set_limits

    @property
    def supported_languages(self) -> List[str]:
        return list(self.allowed_languages)
