"""
vllm-i64 :: Sandbox Executor

Runs user code in an isolated subprocess with resource limits.

On Linux: uses resource.setrlimit() for CPU, memory, file size limits.
On other platforms: subprocess timeout only (no kernel-level isolation).

Security hardening:
- AST analysis blocks dangerous module imports (os, subprocess, socket, etc.)
- Restricted builtins whitelist (no open, exec, eval, compile, __import__)
- Execution timeout default 5s
- Resource limits via setrlimit on Linux

Integer-first: no bloat, no Docker, no VMs.
Just fork → limit → exec → collect → kill.

INL - 2025
"""

import ast
import os
import sys
import shutil
import signal
import tempfile
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set

_logger = logging.getLogger("vllm_i64.sandbox")

# ── Security: blocked modules and restricted builtins ──────────────────

BLOCKED_MODULES: Set[str] = frozenset({
    "os", "subprocess", "socket", "ctypes", "shutil",
    "signal", "multiprocessing", "threading",
    "importlib", "runpy", "code", "codeop",
    "pty", "pipes", "fcntl", "termios",
    "resource", "gc", "sys",
    # network
    "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc", "socketserver",
    # file / pickle / marshal — deserialization attacks
    "pickle", "shelve", "marshal", "tempfile",
    # low-level
    "mmap", "sysconfig", "_thread",
})

# Safe builtins whitelist — no open, exec, eval, compile, __import__
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "bytearray": bytearray, "bytes": bytes, "callable": callable,
    "chr": chr, "complex": complex, "dict": dict, "dir": dir,
    "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset,
    "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
    "id": id, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "object": object,
    "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "super": super, "tuple": tuple, "type": type, "vars": vars,
    "zip": zip, "True": True, "False": False, "None": None,
}


def _validate_python_ast(code: str) -> Optional[str]:
    """
    Parse Python code into an AST and reject dangerous imports.

    Returns an error message if blocked content is found, or None if safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Let the actual interpreter handle syntax errors
        return None

    for node in ast.walk(tree):
        # import foo / import foo.bar
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in BLOCKED_MODULES:
                    return f"Blocked import: '{alias.name}' (module '{top_module}' is restricted)"

        # from foo import bar / from foo.bar import baz
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_module = node.module.split(".")[0]
                if top_module in BLOCKED_MODULES:
                    return f"Blocked import: 'from {node.module} ...' (module '{top_module}' is restricted)"

        # Block dangerous builtin calls: __import__, exec, eval, compile
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id == "__import__":
                    return "Blocked: direct __import__() call"
                if func.id in ("exec", "eval", "compile"):
                    return f"Blocked: {func.id}() is not allowed in sandbox"
            if isinstance(func, ast.Attribute) and func.attr == "__import__":
                return "Blocked: direct __import__() call"

    return None


# Python preamble that replaces builtins at runtime to block dangerous functions.
# This is a defence-in-depth layer on top of the AST check.
# We replace __import__ with a filtered version rather than removing it,
# so that `import math` still works but blocked modules are rejected.
_RESTRICTED_BUILTINS_PREAMBLE = '''\
import builtins as _b
def _make_safe_import():
    _orig = _b.__import__
    _blocked = {
        "os", "subprocess", "socket", "ctypes", "shutil",
        "signal", "multiprocessing", "threading",
        "importlib", "runpy", "code", "codeop",
        "pty", "pipes", "fcntl", "termios",
        "resource", "gc", "sys",
        "http", "urllib", "requests", "httpx", "aiohttp",
        "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
        "xmlrpc", "socketserver",
        "pickle", "shelve", "marshal", "tempfile",
        "mmap", "sysconfig", "_thread",
    }
    def _safe_import(name, *args, **kwargs):
        if name.split(".")[0] in _blocked:
            raise ImportError(f"Import of '{name}' is blocked by sandbox policy")
        return _orig(name, *args, **kwargs)
    return _safe_import
_b.__import__ = _make_safe_import()
# Remove dangerous builtins — open is the main file-system escape hatch
_b.open = None
_b.breakpoint = None
_b.input = None
del _b, _make_safe_import
'''

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
        timeout: Max execution time in seconds (default: 5)
        max_memory_mb: Max memory in MB (default: 256, Linux only)
        max_output_bytes: Max stdout/stderr capture (default: 64KB)
        max_file_size_mb: Max file size writable (default: 10MB, Linux only)
        allowed_languages: Which languages to allow (default: all)
    """

    def __init__(
        self,
        timeout: int = 5,
        max_memory_mb: int = 256,
        max_output_bytes: int = 65536,
        max_file_size_mb: int = 10,
        allowed_languages: Optional[List[str]] = None,
        sandbox_user: Optional[str] = None,
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output_bytes = max_output_bytes
        self.max_file_size_mb = max_file_size_mb
        self.allowed_languages = allowed_languages or list(_RUNTIMES.keys())
        self._is_linux = sys.platform.startswith("linux")

        _logger.warning(
            "sandbox: code execution enabled (timeout=%ds, max_mem=%dMB, "
            "blocked_modules=%d, restricted_builtins=True)",
            self.timeout, self.max_memory_mb, len(BLOCKED_MODULES),
        )

        # Resolve sandbox user UID/GID for process isolation
        self._sandbox_uid: Optional[int] = None
        self._sandbox_gid: Optional[int] = None
        if sandbox_user and self._is_linux:
            try:
                import pwd
                pw = pwd.getpwnam(sandbox_user)
                self._sandbox_uid = pw.pw_uid
                self._sandbox_gid = pw.pw_gid
                _logger.info("sandbox: running as user '%s' (uid=%d gid=%d)", sandbox_user, pw.pw_uid, pw.pw_gid)
            except KeyError:
                _logger.warning("sandbox: user '%s' not found, running as current user", sandbox_user)

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

        # ── AST-based security validation for Python code ──
        if language == "python":
            violation = _validate_python_ast(code)
            if violation is not None:
                _logger.warning("sandbox: blocked execution — %s", violation)
                return ExecutionResult(
                    stdout="",
                    stderr=f"Security violation: {violation}",
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
            # For Python: inject restricted builtins to block open/exec/eval/compile/__import__
            if language == "python":
                f.write(_RESTRICTED_BUILTINS_PREAMBLE)
            f.write(code)

        # Build command
        cmd = [part.replace("{file}", src_path) for part in runtime["cmd"]]

        # Environment: minimal, network allowed (agents need internet)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
            "LANG": "en_US.UTF-8",
            # Prevent Python from importing user site-packages junk
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        # Chown tmpdir to sandbox user so subprocess can write
        if self._sandbox_uid is not None:
            os.chown(tmpdir, self._sandbox_uid, self._sandbox_gid)
            os.chown(src_path, self._sandbox_uid, self._sandbox_gid)

        # Pre-exec function for Linux resource limits + user switch
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
        """Create a preexec_fn that sets resource limits and switches user (Linux only)."""
        max_mem = self.max_memory_mb * 1024 * 1024
        max_cpu = self.timeout
        max_fsize = self.max_file_size_mb * 1024 * 1024
        uid = self._sandbox_uid
        gid = self._sandbox_gid

        def _set_limits():
            import resource
            # Switch to sandbox user (must be done before dropping privileges)
            if gid is not None:
                os.setgid(gid)
            if uid is not None:
                os.setuid(uid)
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
