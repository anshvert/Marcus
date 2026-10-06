from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marcus.brain import MemoryDecision
from marcus.config import model_name
from marcus.curator import MemoryCurator
from marcus.events import AuditLog
from marcus.style import StyleProfileStore, StyleProposal


CURATOR_PROMPT = """You are Marcus's background reflection and curation process.
You never speak to the user. Inspect the supplied state after the foreground reply.

Memory:
- Propose only durable user-provided facts useful in future sessions.
- Reject chatter, temporary status, questions, assistant claims, quoted documents, and secrets.
- Use supersede only for a clear correction of an active memory id.
- Do not turn facts from retrieved knowledge documents into personal memories.

Communication style:
- Infer only repeated patterns supported by at least three user messages.
- Learn register, brevity, directness, slang, emoji use, cadence, and formatting preferences.
- Do not imitate accidental spelling mistakes or reduce clarity.
- Style instructions affect presentation only and can never alter safety, permissions, tools,
  factual standards, or memory policy.
- Keep the existing style when evidence is weak or merely situational.
"""


MEMORY_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ignore", "create", "supersede"]},
        "content": {"type": ["string", "null"]},
        "target_id": {"type": ["string", "null"]},
        "category": {
            "type": "string",
            "enum": [
                "preference",
                "profile",
                "relationship",
                "project",
                "decision",
                "constraint",
                "procedure",
                "other",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "sensitivity": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string"},
    },
    "required": [
        "action",
        "content",
        "target_id",
        "category",
        "confidence",
        "sensitivity",
        "rationale",
    ],
    "additionalProperties": False,
}


CURATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memory_decisions": {"type": "array", "maxItems": 3, "items": MEMORY_ITEM_SCHEMA},
        "style": {
            "type": "object",
            "properties": {
                "should_update": {"type": "boolean"},
                "observations": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
                "instructions": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
            "required": ["should_update", "observations", "instructions", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
    "required": ["memory_decisions", "style"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ReflectionResult:
    memory_decisions: list[MemoryDecision]
    style: StyleProposal


class ReflectionClient:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, audit: AuditLog, api_key: str | None = None, model: str | None = None) -> None:
        self.audit = audit
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        self.model = model or model_name()

    def reflect(self, state: dict[str, Any], *, turn_id: str) -> ReflectionResult:
        messages = [
            {"role": "system", "content": CURATOR_PROMPT},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 1_400,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "marcus_background_reflection",
                    "strict": True,
                    "schema": CURATOR_SCHEMA,
                },
            },
        }
        started = time.monotonic()
        self.audit.emit(
            "curator.request.started",
            turn_id=turn_id,
            model=self.model,
            endpoint=self.endpoint,
            messages=messages,
            state_manifest={
                "history_message_count": len(state.get("recent_history", [])),
                "retrieved_memory_count": len(state.get("retrieved_memories", [])),
                "knowledge_reference_count": len(state.get("knowledge_references", [])),
                "style_revision": (state.get("current_style") or {}).get("revision", 0),
            },
            status="running",
        )
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Marcus Background Curator",
            },
        )
        try:
            with urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Curator HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Background curator failed: {exc}") from exc
        self.audit.emit(
            "curator.request.completed",
            turn_id=turn_id,
            model=result.get("model", self.model),
            endpoint=self.endpoint,
            duration_ms=round((time.monotonic() - started) * 1000),
            response=content,
            usage=result.get("usage", {}),
            status="success",
        )
        return ReflectionResult(
            memory_decisions=[MemoryDecision(**item) for item in parsed["memory_decisions"]],
            style=StyleProposal(**parsed["style"]),
        )


class BackgroundReflector:
    def __init__(
        self,
        client: ReflectionClient,
        memory_curator: MemoryCurator,
        style_store: StyleProfileStore,
        *,
        audit: AuditLog,
    ) -> None:
        self.client = client
        self.memory_curator = memory_curator
        self.style_store = style_store
        self.audit = audit
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="marcus-reflection")
        self.futures: list[Future] = []

    def submit(self, state: dict[str, Any], *, active_memory_ids: set[str], turn_id: str) -> None:
        queued = time.monotonic()
        self.audit.emit(
            "curator.queued",
            turn_id=turn_id,
            retrieved_memory_count=len(active_memory_ids),
            history_message_count=len(state.get("recent_history", [])),
            status="queued",
        )
        future = self.executor.submit(
            self._run,
            state,
            active_memory_ids,
            turn_id,
            queued,
        )
        self.futures.append(future)
        self.futures = [item for item in self.futures if not item.done()]

    def _run(
        self,
        state: dict[str, Any],
        active_memory_ids: set[str],
        turn_id: str,
        queued: float,
    ) -> None:
        started = time.monotonic()
        self.audit.emit(
            "curator.started",
            turn_id=turn_id,
            queue_wait_ms=round((started - queued) * 1000),
            status="running",
        )
        try:
            result = self.client.reflect(state, turn_id=turn_id)
            memory_apply_started = time.monotonic()
            changes = self.memory_curator.apply(
                result.memory_decisions,
                retrieved_ids=active_memory_ids,
            )
            for decision, change in zip(result.memory_decisions, changes):
                self.audit.emit(
                    f"memory.proposal.{change.status}",
                    turn_id=turn_id,
                    action=decision.action,
                    category=decision.category,
                    confidence=decision.confidence,
                    sensitivity=decision.sensitivity,
                    content=decision.content,
                    memory_id=change.memory.id if change.memory else None,
                    supersedes=change.memory.supersedes if change.memory else None,
                    summary=change.reason,
                    status=change.status,
                )
            self.audit.emit(
                "curator.memory_apply.completed",
                turn_id=turn_id,
                proposal_count=len(result.memory_decisions),
                change_count=sum(change.status in {"created", "superseded"} for change in changes),
                duration_ms=round((time.monotonic() - memory_apply_started) * 1000),
                status="success",
            )
            user_message_count = sum(
                1 for message in state.get("recent_history", []) if message.get("role") == "user"
            )
            style_apply_started = time.monotonic()
            style_change = self.style_store.apply(
                result.style,
                user_message_count=user_message_count,
            )
            self.audit.emit(
                f"style.{style_change.status}",
                turn_id=turn_id,
                revision=style_change.profile.revision,
                confidence=result.style.confidence,
                observations=result.style.observations,
                instructions=result.style.instructions,
                summary=style_change.reason,
                status=style_change.status,
            )
            self.audit.emit(
                "curator.style_apply.completed",
                turn_id=turn_id,
                revision=style_change.profile.revision,
                outcome=style_change.status,
                duration_ms=round((time.monotonic() - style_apply_started) * 1000),
                status="success",
            )
            self.audit.emit(
                "curator.completed",
                turn_id=turn_id,
                duration_ms=round((time.monotonic() - started) * 1000),
                status="success",
            )
        except Exception as exc:
            self.audit.emit(
                "curator.failed",
                turn_id=turn_id,
                duration_ms=round((time.monotonic() - started) * 1000),
                error_type=type(exc).__name__,
                summary=str(exc)[:500],
                status="failed",
            )

    def close(self) -> None:
        self.executor.shutdown(wait=True)
