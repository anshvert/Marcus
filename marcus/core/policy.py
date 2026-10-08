from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Risk(IntEnum):
    READ_ONLY = 0
    REVERSIBLE = 1
    EXTERNAL = 2
    DESTRUCTIVE = 3


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    confirmation_required: bool
    reason: str


class ActionPolicy:
    """Initial conservative policy; independent from any language model."""

    def evaluate(self, risk: Risk, *, user_confirmed: bool = False) -> PolicyDecision:
        if risk <= Risk.REVERSIBLE:
            return PolicyDecision(True, False, "Low-risk action")
        if user_confirmed:
            return PolicyDecision(True, False, "Explicit user confirmation recorded")
        return PolicyDecision(
            False,
            True,
            "External and destructive actions require explicit confirmation",
        )


ACTION_RISKS = {
    "ask_jev": Risk.READ_ONLY, "recall_memory": Risk.READ_ONLY,
    "list_memories": Risk.READ_ONLY, "search_knowledge": Risk.READ_ONLY,
    "list_sources": Risk.READ_ONLY, "runtime_status": Risk.READ_ONLY,
    "find_skill": Risk.READ_ONLY, "read_skill": Risk.READ_ONLY,
    "save_memory": Risk.REVERSIBLE, "forget_memory": Risk.REVERSIBLE,
    "delegate_data": Risk.REVERSIBLE, "end_session": Risk.REVERSIBLE,
    "delegate_code": Risk.REVERSIBLE,
}


class CapabilityPolicy:
    """Explicit per-agent capabilities enforced by Python, not model prompts."""

    def __init__(self, allowed: set[str] | None = None) -> None:
        self.allowed = set(ACTION_RISKS) if allowed is None else set(allowed)
        self.risks = ActionPolicy()

    def check(self, action: str) -> PolicyDecision:
        if action not in self.allowed or action not in ACTION_RISKS:
            return PolicyDecision(False, False, "Action is outside this agent's capabilities")
        return self.risks.evaluate(ACTION_RISKS[action])
