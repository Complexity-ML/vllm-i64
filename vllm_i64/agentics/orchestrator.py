"""
vllm-i64 :: Agentics — Multi-Agent Orchestrator

Dispatch multiple agent tasks in parallel against a shared vllm-i64 server.
The server's continuous batching engine handles concurrent requests natively.

Architecture:
    Orchestrator
        ├── TaskQueue (async queue of AgentTask)
        ├── Worker pool (N concurrent Agent instances)
        └── Results collector

Usage:
    orch = Orchestrator(base_url="http://localhost:8000", max_workers=4)

    # Submit tasks
    orch.submit("Write a Python fibonacci function")
    orch.submit("Explain what a KV cache is")
    orch.submit("Find all .py files in /app and count lines")

    # Run all and collect results
    results = asyncio.run(orch.run())
    for r in results:
        print(r.task, "→", r.result[:100])

    # Or use the sync interface:
    results = orch.run_sync()

INL - 2025
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import IntEnum

from .agent import Agent
from .tools import Tool, get_tools

_logger = logging.getLogger("vllm_i64.agentics.orchestrator")


class TaskStatus(IntEnum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3


@dataclass
class AgentTask:
    """A task to be executed by an agent."""
    id: int
    task: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    agent_id: Optional[int] = None


@dataclass
class TaskResult:
    """Result of a completed agent task."""
    task_id: int
    task: str
    result: str
    success: bool
    elapsed_ms: float
    agent_id: int


class Orchestrator:
    """
    Multi-agent orchestrator with parallel task execution.

    Spawns N agent workers that pull tasks from a shared queue.
    All agents hit the same vllm-i64 server, which batches their
    requests together for maximum GPU utilization.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        max_workers: int = 4,
        allow_shell: bool = False,
        max_steps: int = 20,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        verbose: bool = True,
        tools: Optional[Dict[str, Tool]] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.max_workers = max_workers
        self.allow_shell = allow_shell
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.custom_tools = tools

        self._tasks: List[AgentTask] = []
        self._task_counter = 0
        self._results: List[TaskResult] = []

    def submit(self, task: str) -> int:
        """
        Submit a task to the queue. Returns task ID.

        Tasks are not executed until run() or run_sync() is called.
        """
        self._task_counter += 1
        agent_task = AgentTask(id=self._task_counter, task=task)
        self._tasks.append(agent_task)
        if self.verbose:
            print(f"\033[36m[Orchestrator]\033[0m Queued task #{agent_task.id}: {task[:80]}")
        return agent_task.id

    def submit_batch(self, tasks: List[str]) -> List[int]:
        """Submit multiple tasks at once. Returns list of task IDs."""
        return [self.submit(t) for t in tasks]

    def _make_agent(self, agent_id: int) -> Agent:
        """Create a new Agent instance for a worker."""
        return Agent(
            base_url=self.base_url,
            api_key=self.api_key,
            allow_shell=self.allow_shell,
            max_steps=self.max_steps,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=False,  # workers are quiet, orchestrator reports
            tools=self.custom_tools,
        )

    async def _worker(
        self,
        agent_id: int,
        queue: asyncio.Queue,
    ):
        """Worker coroutine: pull tasks from queue and execute them."""
        agent = self._make_agent(agent_id)

        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            task.status = TaskStatus.RUNNING
            task.agent_id = agent_id
            if self.verbose:
                print(f"\033[33m[Worker {agent_id}]\033[0m Starting task #{task.id}: {task.task[:60]}")

            start = time.perf_counter()
            try:
                result = await agent.arun(task.task)
                task.status = TaskStatus.COMPLETED
                task.result = result
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.result = f"Error: {e}"
                _logger.error("Task #%d failed: %s", task.id, e)

            task.elapsed_ms = (time.perf_counter() - start) * 1000

            tr = TaskResult(
                task_id=task.id,
                task=task.task,
                result=task.result or "",
                success=task.status == TaskStatus.COMPLETED,
                elapsed_ms=task.elapsed_ms,
                agent_id=agent_id,
            )
            self._results.append(tr)

            status = "\033[32mOK\033[0m" if tr.success else "\033[31mFAIL\033[0m"
            if self.verbose:
                print(
                    f"\033[33m[Worker {agent_id}]\033[0m Task #{task.id} {status} "
                    f"({task.elapsed_ms:.0f}ms)"
                )

            queue.task_done()

    async def run(self) -> List[TaskResult]:
        """
        Execute all queued tasks with parallel workers.

        Returns list of TaskResult objects.
        """
        if not self._tasks:
            return []

        pending = [t for t in self._tasks if t.status == TaskStatus.PENDING]
        if not pending:
            return self._results

        n_workers = min(self.max_workers, len(pending))
        if self.verbose:
            print(
                f"\033[36m[Orchestrator]\033[0m Running {len(pending)} tasks "
                f"with {n_workers} workers"
            )

        queue: asyncio.Queue = asyncio.Queue()
        for task in pending:
            await queue.put(task)

        start = time.perf_counter()

        workers = [
            asyncio.create_task(self._worker(i + 1, queue))
            for i in range(n_workers)
        ]
        await asyncio.gather(*workers)

        total_ms = (time.perf_counter() - start) * 1000
        succeeded = sum(1 for r in self._results if r.success)
        failed = len(self._results) - succeeded

        if self.verbose:
            print(
                f"\033[36m[Orchestrator]\033[0m Done: {succeeded} succeeded, "
                f"{failed} failed, {total_ms:.0f}ms total"
            )

        return self._results

    def run_sync(self) -> List[TaskResult]:
        """Synchronous wrapper for run()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "Cannot use run_sync() inside an async context. Use 'await orch.run()' instead."
            )
        return asyncio.run(self.run())

    def clear(self):
        """Clear all tasks and results."""
        self._tasks.clear()
        self._results.clear()
        self._task_counter = 0

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._tasks if t.status == TaskStatus.PENDING)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
