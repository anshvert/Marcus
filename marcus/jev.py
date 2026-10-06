from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marcus.events import AuditLog


JEV_MODEL = "typesafe/jev-1.13"
QUESTION_SET = "marcus-turn-routing-v1"

INTENT_CRITERIA = {
    "social": "Greeting, thanks, farewell, casual banter, or small talk with no factual task",
    "general": "A normal question or request answerable without personal files or specialist data work",
    "knowledge": "Question that may require ingested documents, resume, notes, or other source files",
    "memory": "Request about facts, preferences, decisions, or instructions the user previously told Marcus",
    "data_agent": "Request to ingest, import, parse, inspect, transform, analyze, or otherwise work on a file or dataset",
}

ROUTING_QUESTIONS: dict[str, Any] = {
    "intent": {
        "type": "choice",
        "instructions": "Choose the single best route for the user's latest message.",
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
class RouteDecision:
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


class JevRouter:
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
    ) -> RouteDecision:
        deterministic = self._deterministic(text)
        if deterministic:
            self._emit_decision(deterministic, turn_id=turn_id)
            return deterministic
        if not self.api_key:
            decision = self._fallback(text)
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
            "questions": ROUTING_QUESTIONS,
        }
        started = time.monotonic()
        if self.audit:
            self.audit.emit(
                "router.request.started",
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
                "X-OpenRouter-Title": "Marcus Intent Router",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            decision = self._parse(result)
        except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if self.audit:
                self.audit.emit(
                    "router.request.failed",
                    turn_id=turn_id,
                    model=self.model,
                    endpoint=self.endpoint,
                    duration_ms=round((time.monotonic() - started) * 1000),
                    error_type=type(exc).__name__,
                    summary=str(exc)[:240],
                    status="fallback",
                )
            decision = self._fallback(text)
            self._emit_decision(decision, turn_id=turn_id)
            return decision

        if self.audit:
            self.audit.emit(
                "router.request.completed",
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

    def _parse(self, result: dict[str, Any]) -> RouteDecision:
        answers = result["answers"]
        intent_answer = answers["intent"]
        intent = str(intent_answer["choice"])
        if intent not in INTENT_CRITERIA:
            raise ValueError(f"Unknown route returned by JEV: {intent}")
        memory_probability = float(answers["needs_memory"]["noul"])
        knowledge_probability = float(answers["needs_knowledge"]["noul"])
        reflection_probability = float(answers["needs_reflection"]["noul"])
        return RouteDecision(
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
    def _deterministic(text: str) -> RouteDecision | None:
        normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
        greeting_patterns = {
            "hi", "hello", "hey", "yo", "sup", "whats up", "what s up", "whts up",
            "yo whats up", "yo what s up", "yo whts up", "hey whats up", "hey what s up",
            "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye",
        }
        if normalized in greeting_patterns:
            return RouteDecision(intent="social", confidence=1.0, source="deterministic")
        if re.search(r"\b(ingest|import)\b", normalized) and re.search(
            r"\.(pdf|md|markdown)\b", text, flags=re.IGNORECASE
        ):
            return RouteDecision(intent="data_agent", confidence=1.0, source="deterministic")
        return None

    @staticmethod
    def _fallback(text: str) -> RouteDecision:
        normalized = text.casefold()
        if any(word in normalized for word in ("resume", "document", "pdf", "my notes")):
            return RouteDecision(
                intent="knowledge",
                confidence=0.5,
                needs_knowledge=True,
                source="fallback",
            )
        if any(phrase in normalized for phrase in ("remember", "did i tell", "what do you know about me")):
            return RouteDecision(
                intent="memory",
                confidence=0.5,
                needs_memory=True,
                needs_reflection=True,
                source="fallback",
            )
        return RouteDecision(intent="general", confidence=0.5, source="fallback")

    def _emit_decision(self, decision: RouteDecision, *, turn_id: str | None) -> None:
        if self.audit:
            self.audit.emit(
                "router.decision",
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
                status="routed",
            )
