from __future__ import annotations

import json
import time
import uuid
import os
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from marcus.observability.bus import EventBus

_session: ContextVar[str | None] = ContextVar("marcus_session", default=None)


@contextmanager
def session_scope(session_id: str):
    token = _session.set(session_id)
    try:
        yield
    finally:
        _session.reset(token)


def current_session() -> str | None:
    return _session.get()


SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


def _redact(value: Any, key: str = "") -> Any:
    if key.casefold().replace("-", "_") in SENSITIVE_KEYS or key.casefold().endswith("_api_key"):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _redact(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class AuditLog:
    """Append-only structured event log, intentionally separate from chat output."""

    def __init__(self, root: Path = Path("data/logs"), *, session_id: str | None = None, bus: EventBus | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._lock = Lock()
        self.bus = bus or EventBus()
        self._secrets = [value for key, value in os.environ.items()
                         if (key.endswith("API_KEY") or key == "MARCUS_GATEWAY_TOKEN") and len(value) >= 8]

    @property
    def path(self) -> Path:
        day = datetime.now(timezone.utc).date().isoformat()
        return self.root / f"audit-{day}.jsonl"

    def emit(self, event: str, *, turn_id: str | None = None, session_id: str | None = None, **data: Any) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": uuid.uuid4().hex[:12],
            "event": event,
            "session_id": session_id or _session.get() or self.session_id,
            "turn_id": turn_id,
            "data": _redact(data),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for secret in self._secrets:
            line = line.replace(json.dumps(secret)[1:-1], "[REDACTED]")
        record = json.loads(line)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            self.bus.publish(record)
        return record


def latest_audit_path(root: Path = Path("data/logs")) -> Path | None:
    paths = sorted(root.glob("audit-*.jsonl")) if root.exists() else []
    return paths[-1] if paths else None


def read_events(path: Path, *, limit: int = 30) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        lines = deque(stream, maxlen=max(0, limit))
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def follow_events(path: Path, *, from_end: bool = True) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        if from_end:
            stream.seek(0, 2)
        while True:
            line = stream.readline()
            if not line:
                time.sleep(0.2)
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def format_event(record: dict[str, Any], *, include_payloads: bool = False) -> str:
    timestamp = record.get("timestamp", "")
    clock = timestamp[11:19] if len(timestamp) >= 19 else timestamp
    event = record.get("event", "unknown")
    data = record.get("data", {})
    duration = data.get("duration_ms")
    timing = f" +{duration}ms" if duration is not None else ""
    summary = data.get("summary") or data.get("status") or ""
    header = f"{clock}  {event:<34}{timing}"
    if summary:
        header += f"  {summary}"

    details: list[str] = []
    if event in {"model.request.started", "model.request.completed", "curator.request.started", "curator.request.completed", "embedding.request.started", "embedding.request.completed", "jev.request.started", "jev.request.completed"}:
        if data.get("model"):
            details.append(f"model: {data['model']}")
        if data.get("endpoint"):
            details.append(f"endpoint: {data['endpoint']}")
    if event == "turn.received" and data.get("input"):
        details.append(f"user: {data['input']}")
    elif event == "memory.retrieval.completed":
        memories = data.get("memories", [])
        details.extend(
            f"memory [{item.get('id')} · {item.get('kind')}]: {item.get('content')}"
            for item in memories
        )
        if not memories:
            details.append("memory: none retrieved")
    elif event == "embedding.request.started":
        details.append(
            f"batch: {data.get('item_count', len(data.get('inputs', [])))} "
            f"({data.get('input_type', 'unknown')})"
        )
        if include_payloads:
            for index, text in enumerate(data.get("inputs", []), start=1):
                details.append(f"embed[{index}]: {text}")
    elif event == "knowledge.retrieval.completed":
        if include_payloads and data.get("query"):
            details.append(f"query: {data['query']}")
        hits = data.get("hits", [])
        for hit in hits:
            location = f" p.{hit.get('page')}" if hit.get("page") else ""
            text = hit.get("text", "")
            shown = text if include_payloads else _preview(text, 180)
            details.append(
                f"knowledge [{hit.get('title')} / {hit.get('heading')}{location} "
                f"score={hit.get('score')}]: {shown}"
            )
        if not hits:
            details.append("knowledge: none retrieved")
    elif event in {"model.request.started", "curator.request.started"}:
        manifest = data.get("prompt_manifest") or data.get("state_manifest") or {}
        if manifest:
            details.append(f"manifest: {manifest}")
        if include_payloads:
            for message in data.get("messages", []):
                role = message.get("role", "unknown")
                content = message.get("content")
                if content is not None:
                    details.append(f"prompt/{role}: {content}")
                if message.get("tool_calls"):
                    details.append(
                        f"prompt/{role}/tool_calls: "
                        f"{json.dumps(message['tool_calls'], ensure_ascii=False)}"
                    )
                if message.get("tool_call_id"):
                    details.append(f"prompt/{role}/tool_call_id: {message['tool_call_id']}")
            tools = data.get("tools", [])
            if tools:
                details.append(f"tool schemas: {json.dumps(tools, ensure_ascii=False)}")
    elif event in {"model.request.completed", "curator.request.completed"}:
        if data.get("response"):
            details.append(f"response: {data['response']}")
        usage = data.get("usage") or {}
        if usage:
            details.append(f"usage: {usage}")
    elif event == "model.stream.first_output":
        details.append(f"first {data.get('output_type', 'output')} received")
    elif event.startswith("memory.proposal."):
        if data.get("content"):
            details.append(f"candidate: {data['content']}")
        if data.get("memory_id"):
            details.append(f"memory id: {data['memory_id']}")
    elif event in {"memory.created", "memory.archived", "memory.archive.failed"}:
        if data.get("memory_id"):
            details.append(f"memory id: {data['memory_id']}")
        if data.get("content"):
            details.append(f"memory: {data['content']}")
    elif event.startswith("style."):
        if data.get("instructions"):
            details.extend(f"style: {item}" for item in data["instructions"])
        if data.get("observations"):
            details.extend(f"observed: {item}" for item in data["observations"])
    elif event == "curator.started" and data.get("queue_wait_ms") is not None:
        details.append(f"queue wait: {data['queue_wait_ms']}ms")
    elif event == "jev.decision":
        details.append(
            f"intent: {data.get('intent')} via {data.get('source')} "
            f"(confidence={data.get('confidence')})"
        )
        details.append(
            "gates: "
            f"memory={data.get('needs_memory')} ({data.get('memory_probability')}), "
            f"knowledge={data.get('needs_knowledge')} ({data.get('knowledge_probability')}), "
            f"reflection={data.get('needs_reflection')} ({data.get('reflection_probability')})"
        )
    elif event == "jev.request.started" and include_payloads:
        details.append(f"state: {json.dumps(data.get('state', {}), ensure_ascii=False)}")
    elif event == "jev.request.completed":
        usage = data.get("usage") or {}
        if usage:
            details.append(f"usage: {usage}")
        if include_payloads and data.get("answers"):
            details.append(f"answers: {json.dumps(data['answers'], ensure_ascii=False)}")
    elif event == "agent.delegated":
        details.append(
            f"{data.get('from_agent')} → {data.get('to_agent')}: {data.get('task')}"
        )
    elif event.startswith("ingest."):
        keys_by_event = {
            "ingest.started": ("source_path", "destination", "document_id"),
            "ingest.source_copied": ("destination",),
            "ingest.text_extracted": ("destination", "page_count", "character_count"),
            "ingest.chunking.completed": ("chunk_count",),
            "ingest.index_written": ("destination", "embedding_status"),
            "ingest.completed": ("title", "chunk_count", "embedding_status"),
            "ingest.duplicate": ("source_path", "document_id"),
            "ingest.failed": ("source_path", "document_id"),
        }
        for key in keys_by_event.get(event, ()):
            if data.get(key) is not None:
                details.append(f"{key.replace('_', ' ')}: {data[key]}")
    elif event.startswith("tool."):
        if data.get("tool"):
            details.append(f"tool: {data['tool']}")
        if data.get("action"):
            details.append(f"action: {data['action']}")
        if include_payloads and data.get("arguments") is not None:
            details.append(f"arguments: {data['arguments']}")
        if include_payloads and data.get("result"):
            details.append(f"result: {data['result']}")
    elif event.startswith("voice."):
        if data.get("text"):
            details.append(f"speech: {_preview(str(data['text']), 160)}")
        if data.get("engine"):
            details.append(f"engine: {data['engine']}")

    indented_details = ["\n".join(f"    {part}" for part in line.splitlines()) for line in details]
    return "\n".join([header, *indented_details])


def _preview(value: str, limit: int) -> str:
    cleaned = " ".join(value.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"
