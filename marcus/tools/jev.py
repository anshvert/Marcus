from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marcus.observability.audit import AuditLog


JEV_MODEL = "typesafe/jev-1.13"
QUESTION_SET = "marcus-turn-decision-v2"

INTENT_CRITERIA = {
    "casual": "Greeting, thanks, farewell, small talk, playful banter, or a humor request such as asking for a joke",
    "general": "A factual, creative, planning, writing, or reasoning request that needs no personal files or specialist data operation",
    "knowledge": "Question that may require ingested documents, resume, notes, or other source files",
    "memory": "Request about facts, preferences, decisions, or instructions the user previously told Marcus",
    "data_agent": "Request to ingest, import, parse, inspect, transform, or analyze a specified file or dataset using specialist tools",
}

DECISION_QUESTIONS: dict[str, Any] = {
    "intent": {
        "type": "choice",
        "instructions": "Choose the single best intent for the user's latest message.",
        "criteria": INTENT_CRITERIA,
    },
    "needs_memory": {
        "type": "noul",
        "instructions": (
            "Would retrieving personal long-term memories materially help answer this message? "
            "Greetings and self-contained questions should be false."
        ),
    },
    "needs_knowledge": {
        "type": "noul",
        "instructions": (
            "Does this message need facts from an ingested source document such as a resume, "
            "notes, PDF, or Markdown file?"
        ),
    },
    "needs_reflection": {
        "type": "noul",
        "instructions": (
            "Should a background curator inspect this message for a durable personal fact, "
            "preference, correction, decision, constraint, or repeated communication-style evidence? "
            "Pure greetings, transient questions, and file contents should be false."
        ),
    },
}


@dataclass(frozen=True)
class JevDecision:
    intent: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    needs_memory: bool = False
    memory_probability: float = 0.0
    needs_knowledge: bool = False
    knowledge_probability: float = 0.0
    needs_reflection: bool = False
    reflection_probability: float = 0.0
    source: str = "jev"


class JevDecisionTool:
    endpoint = "https://openrouter.ai/api/alpha/decisions"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        audit: AuditLog | None = None,
        threshold: float = 0.65,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("MARCUS_DECISION_MODEL", JEV_MODEL)
        self.audit = audit
        self.threshold = threshold

    def classify(
        self,
        text: str,
        *,
        history: list[dict[str, str]] | None = None,
        turn_id: str | None = None,
    ) -> JevDecision:
        if not self.api_key or os.environ.get("MARCUS_DECISIONS", "1") == "0":
            decision = self._fallback()
            self._emit_decision(decision, turn_id=turn_id)
            return decision

        payload = {
            "model": self.model,
            "state": {
                "latest_user_message": text,
                "prior_user_messages": [
                    message.get("content", "")[:400]
                    for message in (history or [])
                    if message.get("role") == "user"
                ][-4:],
            },
            "questions": DECISION_QUESTIONS,
        }
        started = time.monotonic()
        if self.audit:
            self.audit.emit(
                "jev.request.started",
                turn_id=turn_id,
                model=self.model,
                endpoint=self.endpoint,
                question_set=QUESTION_SET,
                state=payload["state"],
                status="running",
            )
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Marcus JEV Decision Tool",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            decision = self._parse(result)
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if self.audit:
                self.audit.emit(
                    "jev.request.failed",
                    turn_id=turn_id,
                    model=self.model,
                    endpoint=self.endpoint,
                    duration_ms=round((time.monotonic() - started) * 1000),
                    error_type=type(exc).__name__,
                    summary=str(exc)[:240],
                    status="fallback",
                )
            decision = self._fallback()
            self._emit_decision(decision, turn_id=turn_id)
            return decision

        if self.audit:
            self.audit.emit(
                "jev.request.completed",
                turn_id=turn_id,
                model=result.get("model", self.model),
                endpoint=self.endpoint,
                duration_ms=round((time.monotonic() - started) * 1000),
                usage=result.get("usage", {}),
                answers=result.get("answers", {}),
                status="success",
            )
        self._emit_decision(decision, turn_id=turn_id)
        return decision

    def _parse(self, result: dict[str, Any]) -> JevDecision:
        answers = result["answers"]
        intent_answer = answers["intent"]
        intent = str(intent_answer["choice"])
        if intent not in INTENT_CRITERIA:
            raise ValueError(f"Unknown intent returned by JEV: {intent}")
        memory_probability = float(answers["needs_memory"]["noul"])
        knowledge_probability = float(answers["needs_knowledge"]["noul"])
        reflection_probability = float(answers["needs_reflection"]["noul"])
        return JevDecision(
            intent=intent,
            confidence=float(intent_answer.get("confidence", 0.0)),
            probabilities={
                key: float(value) for key, value in intent_answer.get("probabilities", {}).items()
            },
            needs_memory=intent == "memory" or memory_probability >= self.threshold,
            memory_probability=memory_probability,
            needs_knowledge=intent == "knowledge" or knowledge_probability >= self.threshold,
            knowledge_probability=knowledge_probability,
            needs_reflection=reflection_probability >= self.threshold,
            reflection_probability=reflection_probability,
            source="jev",
        )

    @staticmethod
    def _fallback() -> JevDecision:
        return JevDecision(intent="general", confidence=0.5, source="fallback")

    def _emit_decision(self, decision: JevDecision, *, turn_id: str | None) -> None:
        if self.audit:
            self.audit.emit(
                "jev.decision",
                turn_id=turn_id,
                intent=decision.intent,
                confidence=round(decision.confidence, 4),
                probabilities=decision.probabilities,
                needs_memory=decision.needs_memory,
                memory_probability=round(decision.memory_probability, 4),
                needs_knowledge=decision.needs_knowledge,
                knowledge_probability=round(decision.knowledge_probability, 4),
                needs_reflection=decision.needs_reflection,
                reflection_probability=round(decision.reflection_probability, 4),
                source=decision.source,
                status="decided",
            )
