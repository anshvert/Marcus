from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from marcus.knowledge.ingest import DocumentIngestor
from marcus.observability.audit import AuditLog
from marcus.tools.registry import ToolRegistry


@dataclass(frozen=True)
class DataAgentResponse:
    reply: str
    result: dict


class DataAgent:
    """Specialist boundary for ingestion and, later, data analysis operations."""

    name = "data_agent"

    def __init__(self, ingestor: DocumentIngestor, *, audit: AuditLog | None = None) -> None:
        self.audit = audit
        self.tools = ToolRegistry(ingestor, audit=audit)

    def handle(self, request: str, *, turn_id: str | None = None) -> DataAgentResponse:
        started = time.monotonic()
        if self.audit:
            self.audit.emit(
                "agent.delegated",
                turn_id=turn_id,
                from_agent="marcus",
                to_agent=self.name,
                task="ingest_or_data",
                status="delegated",
            )
            self.audit.emit("data_agent.started", turn_id=turn_id, status="running")

        path = self._extract_path(request)
        if not path:
            reply = "Give me the exact PDF or Markdown path you want the data agent to ingest."
            result = {"status": "needs_input", "missing": "path"}
        else:
            document_type = self._document_type(request, path)
            result = self.tools.execute(
                "ingest",
                {"path": path, "label": None, "document_type": document_type},
                turn_id=turn_id,
            )
            if result.get("status") == "failed":
                reply = f"The data agent could not ingest that file: {result['error']}"
            else:
                duplicate = "already indexed" if result.get("duplicate") else "ingested"
                reply = (
                    f"The data agent {duplicate} {result['title']} in "
                    f"{result['chunk_count']} chunks ({result['embedding_status']})."
                )

        if self.audit:
            self.audit.emit(
                "data_agent.completed",
                turn_id=turn_id,
                outcome=result.get("status", "success"),
                duration_ms=round((time.monotonic() - started) * 1000),
                status="success" if result.get("status") != "failed" else "failed",
            )
        return DataAgentResponse(reply=reply, result=result)

    def ingest_path(
        self,
        path: str,
        *,
        label: str | None = None,
        document_type: str = "document",
        turn_id: str | None = None,
    ) -> dict:
        if self.audit:
            self.audit.emit(
                "agent.delegated",
                turn_id=turn_id,
                from_agent="marcus",
                to_agent=self.name,
                task="ingest",
                status="delegated",
            )
        return self.tools.execute(
            "ingest",
            {"path": path, "label": label, "document_type": document_type},
            turn_id=turn_id,
        )

    @staticmethod
    def _extract_path(request: str) -> str | None:
        quoted = re.search(
            r'''["']([^"']+\.(?:pdf|md|markdown))["']''',
            request,
            flags=re.IGNORECASE,
        )
        if quoted:
            return quoted.group(1).strip()
        unquoted = re.search(
            r"((?:/|~/)[^\n]+?\.(?:pdf|md|markdown))(?=\s|$|[,.!?])",
            request,
            flags=re.IGNORECASE,
        )
        return unquoted.group(1).strip() if unquoted else None

    @staticmethod
    def _document_type(request: str, path: str) -> str:
        combined = f"{request} {Path(path).stem}".casefold()
        if "resume" in combined or "cv" in combined:
            return "resume"
        if "profile" in combined:
            return "profile"
        if "note" in combined:
            return "notes"
        return "document"
