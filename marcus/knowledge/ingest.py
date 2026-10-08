from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from marcus.knowledge.embeddings import EmbeddingError, OpenRouterEmbeddings
from marcus.knowledge.store import Chunk, KnowledgeStore
from marcus.observability.audit import AuditLog


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".pdf"}


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    title: str
    document_type: str
    chunk_count: int
    extracted_path: str
    embedding_status: str
    duplicate: bool = False


class DocumentIngestor:
    def __init__(
        self,
        store: KnowledgeStore,
        *,
        vault_path: Path,
        embeddings: OpenRouterEmbeddings | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.store = store
        self.vault_path = vault_path
        self.embeddings = embeddings
        self.audit = audit

    def ingest(
        self,
        path: str,
        *,
        label: str | None = None,
        document_type: str = "document",
        turn_id: str | None = None,
    ) -> IngestResult:
        started = time.monotonic()
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"File does not exist: {source}")
        if source.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported formats currently are PDF and Markdown")
        max_bytes = int(os.environ.get("MARCUS_INGEST_MAX_BYTES", str(25 * 1024 * 1024)))
        if source.stat().st_size > max_bytes:
            raise ValueError("Document exceeds MARCUS_INGEST_MAX_BYTES")
        configured_roots = os.environ.get("MARCUS_INGEST_ROOTS", "")
        if configured_roots:
            roots = [Path(value).expanduser().resolve() for value in configured_roots.split(os.pathsep) if value]
            if not any(source == root or root in source.parents for root in roots):
                raise ValueError("Document is outside the configured ingestion roots")

        source_hash = self._hash(source)
        existing = self.store.find_by_hash(source_hash)
        if existing:
            if self.audit:
                self.audit.emit(
                    "ingest.duplicate",
                    turn_id=turn_id,
                    document_id=existing["id"],
                    source_name=source.name,
                    source_path=str(source),
                    duration_ms=round((time.monotonic() - started) * 1000),
                    status="skipped",
                )
            return IngestResult(
                document_id=existing["id"],
                title=existing["title"],
                document_type=existing["document_type"],
                chunk_count=existing["chunk_count"],
                extracted_path=existing["extracted_path"],
                embedding_status="existing",
                duplicate=True,
            )

        document_id = uuid.uuid4().hex[:12]
        title = (label or source.stem).strip()
        safe_title = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-") or "document"
        destination = self.vault_path / "Marcus" / "Sources" / f"{safe_title}-{source_hash[:8]}"
        destination.mkdir(parents=True, exist_ok=False)
        copied_source = destination / f"original{source.suffix.casefold()}"
        extracted_path = destination / "extracted.md"
        manifest_path = destination / "manifest.json"

        if self.audit:
            self.audit.emit(
                "ingest.started",
                turn_id=turn_id,
                document_id=document_id,
                source_name=source.name,
                source_path=str(source),
                destination=str(destination),
                source_extension=source.suffix.casefold(),
                document_type=document_type,
                status="running",
            )

        try:
            phase_started = time.monotonic()
            shutil.copy2(source, copied_source)
            if self.audit:
                self.audit.emit(
                    "ingest.source_copied",
                    turn_id=turn_id,
                    document_id=document_id,
                    source_path=str(source),
                    destination=str(copied_source),
                    duration_ms=round((time.monotonic() - phase_started) * 1000),
                    status="success",
                )

            phase_started = time.monotonic()
            markdown, page_count = self._extract(source, title)
            extracted_path.write_text(markdown, encoding="utf-8")
            if self.audit:
                self.audit.emit(
                    "ingest.text_extracted",
                    turn_id=turn_id,
                    document_id=document_id,
                    page_count=page_count,
                    character_count=len(markdown),
                    destination=str(extracted_path),
                    duration_ms=round((time.monotonic() - phase_started) * 1000),
                    status="success",
                )

            phase_started = time.monotonic()
            chunks = self._chunk(markdown, document_id)
            if not chunks:
                raise ValueError("No usable text chunks were extracted")
            if self.audit:
                self.audit.emit(
                    "ingest.chunking.completed",
                    turn_id=turn_id,
                    document_id=document_id,
                    chunk_count=len(chunks),
                    duration_ms=round((time.monotonic() - phase_started) * 1000),
                    status="success",
                )

            embedding_status = "not_configured"
            embedding_model: str | None = None
            if self.embeddings:
                try:
                    vectors = self.embeddings.embed(
                        [chunk.text for chunk in chunks],
                        input_type="search_document",
                        turn_id=turn_id,
                    )
                    chunks = [
                        Chunk(
                            id=chunk.id,
                            document_id=chunk.document_id,
                            ordinal=chunk.ordinal,
                            heading=chunk.heading,
                            page=chunk.page,
                            text=chunk.text,
                            embedding=vector,
                        )
                        for chunk, vector in zip(chunks, vectors)
                    ]
                    embedding_status = "indexed"
                    embedding_model = self.embeddings.model
                except EmbeddingError as exc:
                    embedding_status = f"keyword_only: {str(exc)[:180]}"

            imported_at = datetime.now(timezone.utc).isoformat()
            document = {
                "id": document_id,
                "title": title,
                "document_type": document_type,
                "source_name": source.name,
                "source_path": str(copied_source),
                "source_hash": source_hash,
                "extracted_path": str(extracted_path),
                "imported_at": imported_at,
                "metadata": {"page_count": page_count, "extension": source.suffix.casefold()},
            }
            phase_started = time.monotonic()
            self.store.add_document(document, chunks, embedding_model=embedding_model)
            manifest_path.write_text(
                json.dumps(
                    {
                        **document,
                        "chunk_count": len(chunks),
                        "embedding_model": embedding_model,
                        "embedding_status": embedding_status,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if self.audit:
                self.audit.emit(
                    "ingest.index_written",
                    turn_id=turn_id,
                    document_id=document_id,
                    chunk_count=len(chunks),
                    embedding_status=embedding_status,
                    destination=str(self.store.path),
                    duration_ms=round((time.monotonic() - phase_started) * 1000),
                    status="success",
                )
        except Exception:
            if self.audit:
                self.audit.emit(
                    "ingest.failed",
                    turn_id=turn_id,
                    document_id=document_id,
                    source_name=source.name,
                    source_path=str(source),
                    duration_ms=round((time.monotonic() - started) * 1000),
                    status="failed",
                )
            raise

        if self.audit:
            self.audit.emit(
                "ingest.completed",
                turn_id=turn_id,
                document_id=document_id,
                title=title,
                chunk_count=len(chunks),
                embedding_status=embedding_status,
                duration_ms=round((time.monotonic() - started) * 1000),
                status="success",
            )
        return IngestResult(
            document_id=document_id,
            title=title,
            document_type=document_type,
            chunk_count=len(chunks),
            extracted_path=str(extracted_path),
            embedding_status=embedding_status,
        )

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _extract(path: Path, title: str) -> tuple[str, int | None]:
        if path.suffix.casefold() in {".md", ".markdown"}:
            text = path.read_text(encoding="utf-8")
            return f"# {title}\n\n{text.strip()}\n", None

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion requires pypdf. Install Marcus with: pip install -e '.[documents]'"
            ) from exc

        reader = PdfReader(str(path))
        pages: list[str] = [f"# {title}"]
        extracted_chars = 0
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            extracted_chars += len(text)
            pages.append(f"## Page {index}\n\n{text}")
        if extracted_chars < 80:
            raise ValueError(
                "The PDF contains too little extractable text and may be scanned; OCR is not implemented yet"
            )
        return "\n\n".join(pages).strip() + "\n", len(reader.pages)

    @staticmethod
    def _chunk(markdown: str, document_id: str, *, max_chars: int = 1_400) -> list[Chunk]:
        chunks: list[Chunk] = []
        heading = "Document"
        page: int | None = None
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if not text:
                buffer.clear()
                return
            for start in range(0, len(text), max_chars - 180):
                piece = text[start : start + max_chars].strip()
                if not piece:
                    continue
                ordinal = len(chunks)
                chunk_hash = hashlib.sha256(
                    f"{document_id}:{ordinal}:{piece}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        id=chunk_hash,
                        document_id=document_id,
                        ordinal=ordinal,
                        heading=heading,
                        page=page,
                        text=piece,
                    )
                )
            buffer.clear()

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if line.startswith("#"):
                flush()
                heading = line.lstrip("#").strip() or heading
                page_match = re.fullmatch(r"Page (\d+)", heading, flags=re.IGNORECASE)
                page = int(page_match.group(1)) if page_match else page
                continue
            prospective = sum(len(item) + 1 for item in buffer) + len(line)
            if buffer and prospective > max_chars:
                flush()
            if line:
                buffer.append(line)
        flush()
        return chunks
