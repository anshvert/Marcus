from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BANNED_STYLE_TERMS = {
    "permission",
    "tool policy",
    "system prompt",
    "ignore instructions",
    "safety rule",
    "api key",
    "password",
}


@dataclass(frozen=True)
class StyleProfile:
    revision: int
    updated_at: str | None
    sample_count: int
    observations: list[str]
    instructions: list[str]


@dataclass(frozen=True)
class StyleProposal:
    should_update: bool
    observations: list[str]
    instructions: list[str]
    confidence: float
    rationale: str


@dataclass(frozen=True)
class StyleChange:
    status: str
    reason: str
    profile: StyleProfile


class StyleProfileStore:
    """Versioned communication-style overlay; never allowed to change core policy."""

    def __init__(self, vault_path: Path) -> None:
        self.directory = vault_path / "Marcus" / "Behavior"
        self.archive = self.directory / "Archive"
        self.path = self.directory / "style.md"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(self.empty())

    @staticmethod
    def empty() -> StyleProfile:
        return StyleProfile(0, None, 0, [], [])

    def load(self) -> StyleProfile:
        if not self.path.exists():
            return self.empty()
        text = self.path.read_text(encoding="utf-8")
        metadata: dict[str, str] = {}
        if text.startswith("---\n"):
            _, header, body = text.split("---\n", 2)
            for line in header.splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    metadata[key] = value
        else:
            body = text
        observations = self._section(body, "Observed patterns")
        instructions = self._section(body, "Active style instructions")
        return StyleProfile(
            revision=int(metadata.get("revision", "0")),
            updated_at=metadata.get("updated_at") or None,
            sample_count=int(metadata.get("sample_count", "0")),
            observations=observations,
            instructions=instructions,
        )

    def apply(self, proposal: StyleProposal, *, user_message_count: int) -> StyleChange:
        current = self.load()
        if not proposal.should_update:
            return StyleChange("unchanged", proposal.rationale, current)
        if user_message_count < 3:
            return StyleChange("rejected", "At least three user messages are required", current)
        if proposal.confidence < 0.85:
            return StyleChange("rejected", "Style confidence below 0.85", current)
        observations = self._clean(proposal.observations, limit=12)
        instructions = self._clean(proposal.instructions, limit=8)
        if not observations or not instructions:
            return StyleChange("rejected", "Style proposal was empty after validation", current)
        combined = " ".join(instructions).casefold()
        if any(term in combined for term in BANNED_STYLE_TERMS):
            return StyleChange("rejected", "Style cannot modify policy, tools, or secrets", current)

        merged_observations = list(dict.fromkeys([*current.observations, *observations]))[-20:]
        updated = StyleProfile(
            revision=current.revision + 1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            sample_count=max(current.sample_count, user_message_count),
            observations=merged_observations,
            instructions=instructions,
        )
        if self.path.exists() and current.revision > 0:
            shutil.copy2(self.path, self.archive / f"style-v{current.revision}.md")
        self._write(updated)
        return StyleChange("updated", proposal.rationale, updated)

    def prompt_context(self) -> str:
        profile = self.load()
        if not profile.instructions:
            return ""
        return (
            "Learned communication style (presentation only; never overrides safety or tool policy):\n"
            + "\n".join(f"- {item}" for item in profile.instructions)
        )

    def _write(self, profile: StyleProfile) -> None:
        observations = "\n".join(f"- {item}" for item in profile.observations) or "- None yet"
        instructions = "\n".join(f"- {item}" for item in profile.instructions) or "- Use Marcus's default calm, direct style"
        content = (
            "---\n"
            f"revision: {profile.revision}\n"
            f"updated_at: {profile.updated_at or ''}\n"
            f"sample_count: {profile.sample_count}\n"
            "---\n\n"
            "# Observed patterns\n\n"
            f"{observations}\n\n"
            "# Active style instructions\n\n"
            f"{instructions}\n"
        )
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def _section(body: str, heading: str) -> list[str]:
        pattern = rf"(?ms)^# {re.escape(heading)}\s*$\n(.*?)(?=^# |\Z)"
        match = re.search(pattern, body)
        if not match:
            return []
        values = []
        for line in match.group(1).splitlines():
            line = line.strip()
            if line.startswith("- ") and line[2:] not in {"None yet", "Use Marcus's default calm, direct style"}:
                values.append(line[2:].strip())
        return values

    @staticmethod
    def _clean(values: list[str], *, limit: int) -> list[str]:
        cleaned = []
        for value in values[:limit]:
            item = " ".join(value.split()).strip()
            if 5 <= len(item) <= 240:
                cleaned.append(item)
        return cleaned
