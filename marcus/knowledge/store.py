from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

from marcus.knowledge.embeddings import EmbeddingError, OpenRouterEmbeddings
from marcus.observability.audit import AuditLog


TOKEN_RE = re.compile(r"[a-z0-9]+")
SEARCH_STOP_WORDS = {
    "a",
    "about",
    "and",
    "can",
    "did",
    "do",
    "does",
    "document",
    "find",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "section",
    "source",
    "tell",
    "that",
    "the",
    "to",
    "user",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
}
LEXICAL_FAST_PATH_THRESHOLD = 0.25


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    heading: str
    page: int | None
    text: str
    embedding: list[float] | None = None


@dataclass(frozen=True)
class KnowledgeHit:
    chunk_id: str
    document_id: str
    title: str
    heading: str
    page: int | None
    text: str
    score: float


class KnowledgeStore:
    def __init__(
        self,
        path: Path = Path("data/knowledge.sqlite"),
        *,
        embeddings: OpenRouterEmbeddings | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.audit = audit
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL UNIQUE,
                    extracted_path TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    ordinal INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    page INTEGER,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    dimensions INTEGER,
                    embedding_model TEXT
                );
                CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id, ordinal);
                """
            )

    def find_by_hash(self, source_hash: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
        return dict(row) if row else None

    def add_document(self, document: dict, chunks: list[Chunk], *, embedding_model: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, title, document_type, source_name, source_path, source_hash,
                    extracted_path, imported_at, chunk_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["id"],
                    document["title"],
                    document["document_type"],
                    document["source_name"],
                    document["source_path"],
                    document["source_hash"],
                    document["extracted_path"],
                    document["imported_at"],
                    len(chunks),
                    json.dumps(document.get("metadata", {})),
                ),
            )
            for chunk in chunks:
                blob = array("f", chunk.embedding).tobytes() if chunk.embedding else None
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, ordinal, heading, page, text,
                        embedding, dimensions, embedding_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.heading,
                        chunk.page,
                        chunk.text,
                        blob,
                        len(chunk.embedding) if chunk.embedding else None,
                        embedding_model if chunk.embedding else None,
                    ),
                )

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, document_type, source_name, imported_at, chunk_count FROM documents ORDER BY imported_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, *, limit: int = 6, turn_id: str | None = None) -> list[KnowledgeHit]:
        started = time.monotonic()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, d.title FROM chunks c
                JOIN documents d ON d.id = c.document_id
                """
            ).fetchall()
        if not rows:
            if self.audit:
                self.audit.emit(
                    "knowledge.retrieval.completed",
                    turn_id=turn_id,
                    query=query,
                    query_length=len(query),
                    hit_count=0,
                    hits=[],
                    mode="empty_index",
                    duration_ms=round((time.monotonic() - started) * 1000),
                    status="success",
                )
            return []

        all_query_tokens = set(TOKEN_RE.findall(query.casefold()))
        query_tokens = all_query_tokens - SEARCH_STOP_WORDS or all_query_tokens
        lexical_scores: dict[str, float] = {}
        for row in rows:
            searchable = f"{row['title']} {row['heading']} {row['text']}"
            text_tokens = set(TOKEN_RE.findall(searchable.casefold()))
            lexical_scores[row["id"]] = len(query_tokens & text_tokens) / max(
                1,
                len(query_tokens),
            )
        best_lexical = max(lexical_scores.values(), default=0.0)
        use_lexical_fast_path = best_lexical >= LEXICAL_FAST_PATH_THRESHOLD

        query_vector: list[float] | None = None
        if (
            not use_lexical_fast_path
            and self.embeddings
            and any(row["embedding"] is not None for row in rows)
        ):
            try:
                query_vector = self.embeddings.embed(
                    [query],
                    input_type="search_query",
                    turn_id=turn_id,
                )[0]
            except EmbeddingError as exc:
                if self.audit:
                    self.audit.emit(
                        "knowledge.semantic_search.degraded",
                        turn_id=turn_id,
                        summary=str(exc)[:240],
                        status="keyword_fallback",
                    )

        hits: list[KnowledgeHit] = []
        for row in rows:
            lexical = lexical_scores[row["id"]]
            semantic = 0.0
            if query_vector is not None and row["embedding"] is not None:
                vector = array("f")
                vector.frombytes(row["embedding"])
                if len(vector) == len(query_vector):
                    semantic = self._cosine(query_vector, vector)
            score = semantic * 0.85 + lexical * 0.15 if query_vector is not None else lexical
            if score > 0:
                hits.append(
                    KnowledgeHit(
                        chunk_id=row["id"],
                        document_id=row["document_id"],
                        title=row["title"],
                        heading=row["heading"],
                        page=row["page"],
                        text=row["text"],
                        score=score,
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        selected = hits[:limit]
        if self.audit:
            self.audit.emit(
                "knowledge.retrieval.completed",
                turn_id=turn_id,
                query=query,
                query_length=len(query),
                hit_count=len(selected),
                chunk_ids=[item.chunk_id for item in selected],
                scores=[round(item.score, 4) for item in selected],
                hits=[
                    {
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "title": item.title,
                        "heading": item.heading,
                        "page": item.page,
                        "score": round(item.score, 4),
                        "text": item.text,
                    }
                    for item in selected
                ],
                mode=(
                    "hybrid"
                    if query_vector is not None
                    else "keyword_fast_path"
                    if use_lexical_fast_path
                    else "keyword"
                ),
                duration_ms=round((time.monotonic() - started) * 1000),
                status="success",
            )
        return selected

    @staticmethod
    def format_context(hits: list[KnowledgeHit], *, max_chars: int = 8_000) -> str:
        if not hits:
            return "No relevant source-document knowledge was retrieved."
        blocks: list[str] = []
        used = 0
        for hit in hits:
            location = f", page {hit.page}" if hit.page else ""
            block = (
                f"[source: {hit.title}; section: {hit.heading}{location}; "
                f"chunk: {hit.chunk_id}]\n{hit.text}"
            )
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    @staticmethod
    def _cosine(left: list[float], right: array) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
