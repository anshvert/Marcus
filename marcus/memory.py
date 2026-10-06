from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    created_at: str
    kind: str = "explicit"
    source: str = "user"
    supersedes: str | None = None


class MarkdownMemoryStore:
    """User-owned memory stored as individual Obsidian-compatible Markdown notes."""

    def __init__(self, vault_path: Path | None = None) -> None:
        configured = os.environ.get("MARCUS_VAULT_PATH")
        self.vault_path = Path(configured).expanduser() if configured else (
            vault_path or Path("data/vault")
        )
        self.memory_dir = self.vault_path / "Marcus" / "Memory"
        self.archive_dir = self.vault_path / "Marcus" / "Archive"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

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
        self._path(memory.id).write_text(self._serialize(memory), encoding="utf-8")
        return memory

    def list(self, *, limit: int = 20) -> list[Memory]:
        memories = [self._read(path) for path in self.memory_dir.glob("*.md")]
        return sorted(memories, key=lambda item: item.created_at, reverse=True)[:limit]

    def recall(self, query: str, *, limit: int = 5) -> list[Memory]:
        query_tokens = set(TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return []

        ranked: list[tuple[float, Memory]] = []
        for memory in self.list(limit=10_000):
            content_tokens = set(TOKEN_RE.findall(memory.content.lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap:
                score = overlap / len(query_tokens)
                ranked.append((score, memory))

        ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [memory for _, memory in ranked[:limit]]

    def forget(self, memory_id: str) -> bool:
        source = self._path(self._safe_id(memory_id))
        if not source.exists():
            return False
        target = self.archive_dir / source.name
        shutil.move(str(source), str(target))
        return True

    def get(self, memory_id: str) -> Memory | None:
        path = self._path(self._safe_id(memory_id))
        return self._read(path) if path.exists() else None

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        kind: str,
        source: str = "automatic",
    ) -> Memory:
        old_id = self._safe_id(memory_id)
        if self.get(old_id) is None:
            raise ValueError("Cannot supersede a memory that does not exist")
        replacement = self.remember(
            content,
            kind=kind,
            source=source,
            supersedes=old_id,
        )
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
