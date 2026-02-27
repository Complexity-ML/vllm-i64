"""
vllm-i64 :: Structured Logging

Production-grade structured logging with JSON output.
Replaces print() throughout the codebase with proper log levels,
request tracing, and machine-parseable output.

INL - 2025
"""

import logging
import json
import time
import sys
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = self.formatTime(record, "%H:%M:%S")
        prefix = f"{color}{ts} [{record.levelname:>7}]{self.RESET}"
        msg = f"{prefix} {record.getMessage()}"
        if hasattr(record, "request_id"):
            msg += f" [req={record.request_id}]"
        if record.exc_info and record.exc_info[1]:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured logging for vllm-i64.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Use JSON format (for production)
        log_file: Optional file path for log output
    """
    logger = logging.getLogger("vllm_i64")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = JSONFormatter() if json_output else HumanFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JSONFormatter())  # Always JSON for files
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "vllm_i64") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class RequestLogger:
    """Context manager for request-scoped logging."""

    def __init__(self, request_id: int, logger: Optional[logging.Logger] = None):
        self.request_id = request_id
        self.logger = logger or get_logger()
        self.start_time = time.perf_counter()

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra={"request_id": self.request_id, "extra_data": kwargs})

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra={"request_id": self.request_id, "extra_data": kwargs})

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={"request_id": self.request_id, "extra_data": kwargs})

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000
