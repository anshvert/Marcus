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
