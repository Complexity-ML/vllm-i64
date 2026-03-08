"""Lightweight in-process event bus for agent observability.

Events are emitted by sandbox, RAG, and completion handlers, then
consumed by SSE clients connected to /v1/agent/events.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger("vllm_i64.events")


@dataclass
class AgentEvent:
    """Single event in an agent session."""
    type: str                           # "sandbox", "rag_search", "rag_index", "completion", "error"
    session_id: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Fan-out event bus: multiple SSE subscribers receive all events."""

    def __init__(self, history_limit: int = 200):
        self._subscribers: dict[str, asyncio.Queue[AgentEvent | None]] = {}
        self._history: list[AgentEvent] = []
        self._history_limit = history_limit

    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all subscribers (non-blocking)."""
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        for sub_id, queue in list(self._subscribers.items()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Subscriber %s queue full, dropping event", sub_id)

    def subscribe(self, session_filter: str | None = None) -> tuple[str, asyncio.Queue[AgentEvent | None]]:
        """Create a new subscriber. Returns (subscriber_id, queue).

        If session_filter is set, only events for that session are delivered.
        Filtering is done at iteration time by the SSE handler.
        """
        sub_id = uuid.uuid4().hex[:8]
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=500)
        self._subscribers[sub_id] = queue
        logger.info("New subscriber: %s (filter=%s)", sub_id, session_filter)
        return sub_id, queue

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscriber."""
        self._subscribers.pop(sub_id, None)
        logger.info("Unsubscribed: %s", sub_id)

    def get_history(self, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent events, optionally filtered by session."""
        events = self._history
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        return [e.to_dict() for e in events[-limit:]]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
