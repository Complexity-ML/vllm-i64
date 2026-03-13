"""Tests for sandbox executor security restrictions."""

import pytest
from vllm_i64.sandbox.executor import (
    Sandbox,
    ExecutionResult,
    _validate_python_ast,
    BLOCKED_MODULES,
)


# ── AST validation tests ──────────────────────────────────────────────


class TestASTValidation:
    """Test _validate_python_ast blocks dangerous imports."""

    def test_safe_code_passes(self):
        assert _validate_python_ast("x = 1 + 2\nprint(x)") is None

    def test_import_math_passes(self):
        assert _validate_python_ast("import math\nprint(math.pi)") is None

    def test_import_json_passes(self):
        assert _validate_python_ast("import json") is None

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "socket", "ctypes", "shutil", "sys",
        "signal", "multiprocessing", "pickle", "marshal",
        "importlib", "tempfile", "mmap",
    ])
    def test_blocked_direct_import(self, module):
        result = _validate_python_ast(f"import {module}")
        assert result is not None
        assert "Blocked import" in result

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "socket", "ctypes", "shutil",
    ])
    def test_blocked_from_import(self, module):
        result = _validate_python_ast(f"from {module} import *")
        assert result is not None
        assert "Blocked import" in result

    def test_blocked_submodule_import(self):
        result = _validate_python_ast("import os.path")
        assert result is not None
        assert "os" in result

    def test_blocked_from_submodule(self):
        result = _validate_python_ast("from os.path import join")
        assert result is not None

    def test_blocked_dunder_import_call(self):
        result = _validate_python_ast('__import__("os")')
        assert result is not None
        assert "__import__" in result

    def test_blocked_http(self):
        result = _validate_python_ast("import http.client")
        assert result is not None

    def test_blocked_urllib(self):
        result = _validate_python_ast("from urllib.request import urlopen")
        assert result is not None

    def test_syntax_error_passes_through(self):
        # Syntax errors should not be caught by AST validation
        # (let the interpreter report them)
        assert _validate_python_ast("def def def") is None


# ── Sandbox integration tests ─────────────────────────────────────────


class TestSandboxSecurity:
    """Test the Sandbox class enforces security restrictions."""

    @pytest.fixture
    def sandbox(self):
        return Sandbox(timeout=5)

    def test_default_timeout_is_5(self):
        s = Sandbox()
        assert s.timeout == 5

    def test_safe_code_executes(self, sandbox):
        result = sandbox.execute("print('hello')")
        assert result.success
        assert "hello" in result.stdout

    def test_blocked_import_os(self, sandbox):
        result = sandbox.execute("import os\nprint(os.getcwd())")
        assert not result.success
        assert "Security violation" in result.stderr
        assert "os" in result.stderr

    def test_blocked_import_subprocess(self, sandbox):
        result = sandbox.execute("import subprocess\nsubprocess.run(['ls'])")
        assert not result.success
        assert "Security violation" in result.stderr

    def test_blocked_import_socket(self, sandbox):
        result = sandbox.execute("import socket")
        assert not result.success
        assert "Security violation" in result.stderr

    def test_blocked_dunder_import(self, sandbox):
        result = sandbox.execute('m = __import__("os")')
        assert not result.success
        assert "Security violation" in result.stderr

    def test_blocked_from_os_import(self, sandbox):
        result = sandbox.execute("from os import system\nsystem('whoami')")
        assert not result.success
        assert "Security violation" in result.stderr

    def test_blocked_open_at_runtime(self, sandbox):
        # open() is neutered by the builtins preamble at runtime
        result = sandbox.execute("f = open('/etc/passwd')")
        assert not result.success

    def test_blocked_eval_by_ast(self, sandbox):
        result = sandbox.execute("eval('1+1')")
        assert not result.success
        assert "Security violation" in result.stderr

    def test_blocked_exec_by_ast(self, sandbox):
        result = sandbox.execute("exec('x=1')")
        assert not result.success
        assert "Security violation" in result.stderr

    def test_blocked_compile_by_ast(self, sandbox):
        result = sandbox.execute("compile('x=1', '<string>', 'exec')")
        assert not result.success
        assert "Security violation" in result.stderr

    def test_language_not_allowed(self, sandbox):
        s = Sandbox(allowed_languages=["python"])
        result = s.execute("console.log('hi')", language="node")
        assert not result.success
        assert "not allowed" in result.stderr

    def test_safe_math_works(self, sandbox):
        result = sandbox.execute("import math\nprint(math.sqrt(144))")
        assert result.success
        assert "12" in result.stdout

    def test_safe_json_works(self, sandbox):
        result = sandbox.execute('import json\nprint(json.dumps({"a": 1}))')
        assert result.success
        assert '"a"' in result.stdout

    def test_exit_code_nonzero_on_error(self, sandbox):
        result = sandbox.execute("raise ValueError('boom')")
        assert not result.success
        assert result.exit_code != 0


class TestExecutionResult:
    def test_success_property(self):
        r = ExecutionResult(stdout="ok", stderr="", exit_code=0)
        assert r.success

    def test_failure_property(self):
        r = ExecutionResult(stdout="", stderr="err", exit_code=1)
        assert not r.success

    def test_timeout_not_success(self):
        r = ExecutionResult(stdout="", stderr="", exit_code=0, timed_out=True)
        assert not r.success

    def test_to_dict(self):
        r = ExecutionResult(stdout="hi", stderr="", exit_code=0, language="python")
        d = r.to_dict()
        assert d["success"] is True
        assert d["stdout"] == "hi"

    def test_to_tool_output_no_output(self):
        r = ExecutionResult(stdout="", stderr="", exit_code=0)
        assert r.to_tool_output() == "(no output)"
