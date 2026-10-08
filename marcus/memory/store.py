from __future__ import annotations

import os
import re
import shutil
import uuid
import sqlite3
import threading
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
QUERY_STOP_WORDS = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "can",
    "did",
    "do",
    "does",
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
    "the",
    "tell",
    "that",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
}


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    created_at: str
    kind: str = "explicit"
    source: str = "user"
    supersedes: str | None = None


class MarkdownMemoryStore:
    """Markdown source of truth, daily fact files, and a rebuildable SQLite index.

    Existing individual notes remain readable. New facts share a daily file;
    forgetting exports the removed fact to its own recoverable archive note.
    """

    def __init__(self, vault_path: Path | None = None) -> None:
        configured = os.environ.get("MARCUS_VAULT_PATH")
        self.vault_path = Path(configured).expanduser() if configured else (
            vault_path or Path("data/vault")
        )
        self.memory_dir = self.vault_path / "Marcus" / "Memory"
        self.archive_dir = self.vault_path / "Marcus" / "Archive"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir = self.memory_dir / "Daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vault_path / "Marcus" / "memory.sqlite"
        self._lock = threading.RLock()
        self._snapshot: dict[str, tuple[int, int]] = {}
        self.sync_errors: dict[str, str] = {}
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT, created_at TEXT, kind TEXT, source TEXT, supersedes TEXT, path TEXT)")
            connection.execute("CREATE INDEX IF NOT EXISTS memory_created ON memories(created_at)")
            connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, content)")
        self.sync()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.index_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".marcus-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def sync(self) -> None:
        """Index manual Obsidian edits without rewriting the source notes."""
        with self._lock:
            files = list(self.memory_dir.glob("*.md")) + list(self.daily_dir.glob("*.md"))
            snapshot = {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in files}
            with self._connect() as connection:
                indexed_paths = {row[0] for row in connection.execute("SELECT DISTINCT path FROM memories")}
                removed = indexed_paths - snapshot.keys()
                changed = [path for path in files if self._snapshot.get(str(path)) != snapshot[str(path)]]
                for path in [*removed, *map(str, changed)]:
                    ids = [row[0] for row in connection.execute("SELECT id FROM memories WHERE path=?", (path,))]
                    for memory_id in ids:
                        connection.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
                    connection.execute("DELETE FROM memories WHERE path=?", (path,))
                for path in changed:
                    self.sync_errors.pop(str(path), None)
                    try:
                        for memory in self._read_many(path):
                            self._safe_id(memory.id)
                            connection.execute("DELETE FROM memory_fts WHERE id=?", (memory.id,))
                            connection.execute("INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?)",
                                               (memory.id, memory.content, memory.created_at, memory.kind,
                                                memory.source, memory.supersedes, str(path)))
                            connection.execute("INSERT INTO memory_fts VALUES (?,?)", (memory.id, memory.content))
                    except (ValueError, KeyError, UnicodeError) as exc:
                        self.sync_errors[str(path)] = str(exc)[:240]
            self._snapshot = snapshot
            if changed or removed:
                self._refresh_layers()

    def _refresh_layers(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT 100").fetchall()
        memories = [self._row(row) for row in rows]
        profile = [item for item in memories if item.kind in {"profile", "preference", "constraint"}]
        for filename, title, items in [("USER.md", "User profile", profile), ("MEMORY.md", "Durable memory overview", memories)]:
            context = self.format_context(items, max_chars=4000)
            path = self.vault_path / "Marcus" / filename
            if path.exists() and "Generated from active memories;" not in path.read_text(encoding="utf-8"):
                continue
            self._atomic_write(path,
                               f"# {title}\n\nGenerated from active memories; edit the source notes in Memory/ to change facts.\n\n{context}\n")

    def core_context(self, *, max_chars: int = 1000) -> str:
        self.sync()
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories WHERE kind IN ('profile','preference','constraint') ORDER BY created_at DESC LIMIT 12").fetchall()
        memories = [self._row(row) for row in rows]
        return self.format_context(memories, max_chars=max_chars) if memories else ""

    @staticmethod
    def _row(row) -> Memory:
        return Memory(**{key: row[key] for key in Memory.__dataclass_fields__})

    def remember(
        self,
        content: str,
        *,
        kind: str = "explicit",
        source: str = "user",
        supersedes: str | None = None,
    ) -> Memory:
        cleaned = " ".join(content.split()).strip()
        if not cleaned:
            raise ValueError("Memory content cannot be empty")

        memory = Memory(
            id=uuid.uuid4().hex[:12],
            content=cleaned,
            created_at=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            source=source,
            supersedes=supersedes,
        )
        with self._lock:
            path = self.daily_dir / f"{memory.created_at[:10]}.md"
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            if existing:
                self._read_many(path)  # Reject malformed daily notes before appending.
            self._atomic_write(path, existing + self._serialize(memory) + "\n")
            self.sync()
        return memory

    def list(self, *, limit: int = 20) -> list[Memory]:
        self.sync()
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (max(0, limit),)).fetchall()
        return [self._row(row) for row in rows]

    def recall(self, query: str, *, limit: int = 5) -> list[Memory]:
        query_tokens = set(TOKEN_RE.findall(query.lower())) - QUERY_STOP_WORDS
        if not query_tokens:
            return []

        self.sync()
        expression = " OR ".join('"' + token + '"' for token in sorted(query_tokens))
        with self._connect() as connection:
            rows = connection.execute("SELECT m.* FROM memories m JOIN memory_fts f ON m.id=f.id WHERE memory_fts MATCH ? ORDER BY bm25(memory_fts), m.created_at DESC LIMIT 200", (expression,)).fetchall()
        ranked: list[tuple[float, Memory]] = []
        for row in rows:
            memory = self._row(row)
            content_tokens = set(TOKEN_RE.findall(memory.content.lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap:
                score = overlap / len(query_tokens)
                ranked.append((score, memory))

        ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [memory for _, memory in ranked[:limit]]

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            self.sync()
            with self._connect() as connection:
                row = connection.execute("SELECT * FROM memories WHERE id=?", (self._safe_id(memory_id),)).fetchone()
            if row is None:
                return False
            memory = self._row(row)
            source = Path(row["path"])
            target = self.archive_dir / f"{memory.id}.md"
            self._atomic_write(target, self._serialize(memory))
            if source.parent == self.daily_dir:
                remaining = [item for item in self._read_many(source) if item.id != memory.id]
                self._atomic_write(source, "\n".join(self._serialize(item) for item in remaining))
            else:
                shutil.move(str(source), str(target))
            self.sync()
            return True

    def get(self, memory_id: str) -> Memory | None:
        self.sync()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id=?", (self._safe_id(memory_id),)).fetchone()
        return self._row(row) if row else None

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        kind: str,
        source: str = "automatic",
    ) -> Memory:
        with self._lock:
            old_id = self._safe_id(memory_id)
            if self.get(old_id) is None:
                raise ValueError("Cannot supersede a memory that does not exist")
            replacement = self.remember(content, kind=kind, source=source, supersedes=old_id)
            self.forget(old_id)
            return replacement

    @staticmethod
    def format_context(memories: list[Memory], *, max_chars: int = 4_000) -> str:
        if not memories:
            return "No relevant long-term memories were retrieved."
        lines: list[str] = []
        used = 0
        for memory in memories:
            line = f"- [{memory.id}] ({memory.kind}) {memory.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "Relevant user-owned memories:\n" + "\n".join(lines)

    def _path(self, memory_id: str) -> Path:
        return self.memory_dir / f"{self._safe_id(memory_id)}.md"

    @staticmethod
    def _safe_id(memory_id: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{12}", memory_id):
            raise ValueError("Invalid memory id")
        return memory_id

    @staticmethod
    def _serialize(memory: Memory) -> str:
        supersedes = f"supersedes: {memory.supersedes}\n" if memory.supersedes else ""
        return (
            "---\n"
            f"id: {memory.id}\n"
            f"created_at: {memory.created_at}\n"
            f"kind: {memory.kind}\n"
            f"source: {memory.source}\n"
            f"{supersedes}"
            "tags:\n"
            "  - marcus/memory\n"
            "---\n\n"
            f"{memory.content}\n"
        )

    @staticmethod
    def _read(path: Path) -> Memory:
        text = path.read_text(encoding="utf-8")
        return MarkdownMemoryStore._parse(text, path)

    @staticmethod
    def _read_many(path: Path) -> list[Memory]:
        text = path.read_text(encoding="utf-8")
        if path.parent.name != "Daily":
            return [MarkdownMemoryStore._parse(text, path)]
        return [MarkdownMemoryStore._parse(block, path) for block in re.split(r"(?m)(?=^---\nid: )", text) if block.strip()]

    @staticmethod
    def _parse(text: str, path: Path) -> Memory:
        if not text.startswith("---\n"):
            raise ValueError(f"Invalid Marcus memory note: {path}")

        _, header, body = text.split("---\n", 2)
        metadata: dict[str, str] = {}
        for line in header.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                metadata[key] = value

        return Memory(
            id=metadata["id"],
            content=body.strip(),
            created_at=metadata["created_at"],
            kind=metadata.get("kind", "explicit"),
            source=metadata.get("source", "user"),
            supersedes=metadata.get("supersedes"),
        )
