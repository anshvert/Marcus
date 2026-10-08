from __future__ import annotations

import copy
import queue
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator, TypedDict


class EventRecord(TypedDict):
    schema_version: int
    timestamp: str
    event_id: str
    event: str
    session_id: str
    turn_id: str | None
    data: dict[str, Any]


class EventBus:
    """Bounded, non-blocking fan-out. Slow clients cannot stall the agent."""

    def __init__(self, *, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._recent: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def publish(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._recent.append(copy.deepcopy(record))
            for subscriber in self._subscribers:
                if subscriber.full():
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        pass
                subscriber.put_nowait(copy.deepcopy(record))

    @contextmanager
    def subscribe(self, *, replay: int = 0) -> Iterator[queue.Queue]:
        subscriber: queue.Queue = queue.Queue(maxsize=self.capacity)
        with self._lock:
            for record in list(self._recent)[-min(replay, self.capacity):] if replay else []:
                subscriber.put_nowait(copy.deepcopy(record))
            self._subscribers.add(subscriber)
        try:
            yield subscriber
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)
