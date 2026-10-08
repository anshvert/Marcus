from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from marcus.knowledge.ingest import DocumentIngestor
from marcus.observability.audit import AuditLog


INGEST_TOOL = {
    "type": "function",
    "function": {
        "name": "ingest",
        "description": (
            "Import a user-specified local PDF or Markdown file into the data agent's knowledge base. "
            "Call only when the user explicitly asks to ingest/import a file and supplies its path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Exact local filesystem path supplied by the user"},
                "label": {"type": ["string", "null"], "description": "Human-readable document title"},
                "document_type": {
                    "type": "string",
                    "enum": ["resume", "profile", "notes", "reference", "document"],
                },
            },
            "required": ["path", "label", "document_type"],
            "additionalProperties": False,
        },
    },
}


class ToolRegistry:
    def __init__(self, ingestor: DocumentIngestor, *, audit: AuditLog | None = None) -> None:
        self.ingestor = ingestor
        self.audit = audit

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [INGEST_TOOL]

    def execute(self, name: str, arguments: dict[str, Any], *, turn_id: str | None = None) -> dict:
        started = time.monotonic()
        if self.audit:
            self.audit.emit(
                "tool.execution.started",
                turn_id=turn_id,
                tool=name,
                arguments=arguments,
                status="running",
            )
        try:
            if name != "ingest":
                raise ValueError(f"Unknown tool: {name}")
            result = self.ingestor.ingest(
                arguments["path"],
                label=arguments.get("label"),
                document_type=arguments.get("document_type", "document"),
                turn_id=turn_id,
            )
            output = asdict(result)
            if self.audit:
                self.audit.emit(
                    "tool.execution.completed",
                    turn_id=turn_id,
                    tool=name,
                    document_id=result.document_id,
                    result=output,
                    duration_ms=round((time.monotonic() - started) * 1000),
                    status="success",
                )
            return output
        except Exception as exc:
            if self.audit:
                self.audit.emit(
                    "tool.execution.failed",
                    turn_id=turn_id,
                    tool=name,
                    error_type=type(exc).__name__,
                    summary=str(exc)[:240],
                    duration_ms=round((time.monotonic() - started) * 1000),
                    status="failed",
                )
            return {"status": "failed", "error": str(exc)}
