from __future__ import annotations

from dataclasses import dataclass

from marcus.brain import MemoryDecision
from marcus.memory import MarkdownMemoryStore, Memory


ALLOWED_CATEGORIES = {
    "preference",
    "profile",
    "relationship",
    "project",
    "decision",
    "constraint",
    "procedure",
}


@dataclass(frozen=True)
class MemoryChange:
    status: str
    reason: str
    memory: Memory | None = None


class MemoryCurator:
    """Deterministic gate between model suggestions and durable memory."""

    def __init__(self, store: MarkdownMemoryStore, *, threshold: float = 0.85) -> None:
        self.store = store
        self.threshold = threshold

    def apply(
        self,
        decisions: list[MemoryDecision],
        *,
        retrieved_ids: set[str],
    ) -> list[MemoryChange]:
        return [self._apply_one(item, retrieved_ids=retrieved_ids) for item in decisions]

    def _apply_one(self, decision: MemoryDecision, *, retrieved_ids: set[str]) -> MemoryChange:
        if decision.action == "ignore":
            return MemoryChange("rejected", "Model judged it non-durable")
        if decision.category not in ALLOWED_CATEGORIES:
            return MemoryChange("rejected", "Unsupported memory category")
        if decision.sensitivity == "high":
            return MemoryChange("rejected", "High-sensitivity information is not auto-stored")
        if decision.confidence < self.threshold:
            return MemoryChange("rejected", "Confidence below automatic-storage threshold")

        content = " ".join((decision.content or "").split()).strip()
        if not 10 <= len(content) <= 500:
            return MemoryChange("rejected", "Memory must contain a concise durable fact")

        normalized = content.casefold()
        if any(item.content.casefold() == normalized for item in self.store.list(limit=10_000)):
            return MemoryChange("rejected", "Duplicate active memory")

        if decision.action == "create":
            memory = self.store.remember(
                content,
                kind=decision.category,
                source="automatic",
            )
            return MemoryChange("created", decision.rationale, memory)

        if decision.action == "supersede":
            target = decision.target_id
            if not target or target not in retrieved_ids:
                return MemoryChange("rejected", "Supersede target was not retrieved for this turn")
            if self.store.get(target) is None:
                return MemoryChange("rejected", "Supersede target is not an active memory")
            memory = self.store.supersede(
                target,
                content,
                kind=decision.category,
                source="automatic",
            )
            return MemoryChange("superseded", decision.rationale, memory)

        return MemoryChange("rejected", "Unknown memory action")
