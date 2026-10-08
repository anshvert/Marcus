from __future__ import annotations

import time
import copy
from threading import Event
from dataclasses import asdict
from typing import Any

from marcus.agents.data import DataAgent
from marcus.agents.registry import SpecialistRegistry
from marcus.core.config import model_name
from marcus.core.policy import CapabilityPolicy
from marcus.skills.store import SkillStore
from marcus.knowledge.store import KnowledgeHit, KnowledgeStore
from marcus.memory.store import MarkdownMemoryStore, Memory
from marcus.observability.audit import AuditLog
from marcus.tools.jev import JevDecision, JevDecisionTool


MARCUS_ACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "marcus_action",
        "description": (
            "Act only when context or an operation is needed; otherwise answer directly. "
            "ask_jev is only for ambiguity that changes routing. save_memory and end_session "
            "require an explicit user request. recall_memory searches facts learned from chat; "
            "search_knowledge searches ingested documents. delegate_data handles file work. "
            "find_skill/read_skill discover/load workflows. delegate_code uses the enabled coding specialist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "ask_jev",
                        "delegate_data",
                        "recall_memory",
                        "list_memories",
                        "save_memory",
                        "forget_memory",
                        "search_knowledge",
                        "list_sources",
                        "runtime_status",
                        "end_session",
                        "find_skill",
                        "read_skill",
                        "delegate_code",
                    ],
                },
                "input": {
                    "type": ["string", "null"],
                    "description": "Request, query, content/id, or null.",
                },
            },
            "required": ["action", "input"],
            "additionalProperties": False,
        },
    },
}


class MarcusToolbox:
    """Single compact tool surface used by the conversational manager."""

    def __init__(
        self,
        *,
        decision_tool: JevDecisionTool,
        data_agent: DataAgent,
        memory: MarkdownMemoryStore,
        knowledge: KnowledgeStore,
        audit: AuditLog,
        policy: CapabilityPolicy | None = None,
        skills: SkillStore | None = None,
        cancel: Event | None = None,
        specialists: SpecialistRegistry | None = None,
    ) -> None:
        self.decision_tool = decision_tool
        self.data_agent = data_agent
        self.memory = memory
        self.knowledge = knowledge
        self.audit = audit
        self.policy = policy or CapabilityPolicy()
        self.skills = skills or SkillStore()
        self.cancel = cancel
        self.specialists = specialists
        self.user_text = ""
        self.history: list[dict[str, str]] = []
        self.retrieved_memories: list[Memory] = []
        self.knowledge_hits: list[KnowledgeHit] = []
        self.jev_decision: JevDecision | None = None
        self.end_session_requested = False
        self.suppress_reflection = False

    @property
    def schemas(self) -> list[dict[str, Any]]:
        schema = copy.deepcopy(MARCUS_ACTION_TOOL)
        actions = schema["function"]["parameters"]["properties"]["action"]["enum"]
        if not self.specialists or not self.specialists.has("coding_agent"):
            actions.remove("delegate_code")
        schema["function"]["parameters"]["properties"]["action"]["enum"] = [item for item in actions if self.policy.check(item).allowed]
        return [schema] if schema["function"]["parameters"]["properties"]["action"]["enum"] else []

    def begin_turn(self, user_text: str, history: list[dict[str, str]]) -> None:
        self.user_text = user_text
        self.history = list(history)
        self.retrieved_memories = []
        self.knowledge_hits = []
        self.jev_decision = None
        self.end_session_requested = False
        self.suppress_reflection = False

    def source_catalog(self) -> str:
        documents = self.knowledge.list_documents()
        if not documents:
            return "No ingested documents are available."
        lines = [
            f"- {str(item['title'])[:160]} ({str(item['document_type'])[:40]}; source: {str(item['source_name'])[:160]})"
            for item in documents[:30]
        ]
        return ("Available ingested documents (titles only; contents not loaded):\n" + "\n".join(lines))[:4000]

    def execute(self, name: str, arguments: dict[str, Any], *, turn_id: str | None = None) -> dict:
        started = time.monotonic()
        action = str(arguments.get("action", "")) if isinstance(arguments, dict) else ""
        value = arguments.get("input") if isinstance(arguments, dict) else None
        self.audit.emit(
            "tool.execution.started",
            turn_id=turn_id,
            tool=name,
            action=action,
            arguments=arguments,
            status="running",
        )
        try:
            if self.cancel is not None and self.cancel.is_set():
                self.audit.emit("tool.execution.cancelled", turn_id=turn_id, tool=name, action=action,
                                duration_ms=round((time.monotonic() - started) * 1000), status="cancelled")
                return {"status": "cancelled"}
            if name != "marcus_action":
                raise ValueError(f"Unknown manager tool: {name}")
            if not isinstance(arguments, dict) or set(arguments) != {"action", "input"} or not isinstance(arguments["action"], str) or not (value is None or isinstance(value, str)):
                raise ValueError("Arguments do not match the Marcus action schema")
            if value is not None and len(value) > 8000:
                raise ValueError("Action input exceeds the size limit")
            decision = self.policy.check(action)
            self.audit.emit("policy.evaluated", turn_id=turn_id, action=action,
                            allowed=decision.allowed, summary=decision.reason, status="allowed" if decision.allowed else "denied")
            if not decision.allowed:
                self.audit.emit("tool.execution.denied", turn_id=turn_id, tool=name, action=action,
                                duration_ms=round((time.monotonic() - started) * 1000), status="denied")
                return {"status": "denied", "reason": decision.reason}
            result = self._execute_action(action, value, turn_id=turn_id)
            self.audit.emit(
                "tool.execution.completed",
                turn_id=turn_id,
                tool=name,
                action=action,
                result=result,
                duration_ms=round((time.monotonic() - started) * 1000),
                status=result.get("status", "success"),
            )
            return result
        except Exception as exc:
            self.audit.emit(
                "tool.execution.failed",
                turn_id=turn_id,
                tool=name,
                action=action,
                error_type=type(exc).__name__,
                summary=str(exc)[:240],
                duration_ms=round((time.monotonic() - started) * 1000),
                status="failed",
            )
            return {"status": "failed", "error": str(exc)}

    def _execute_action(self, action: str, value: Any, *, turn_id: str | None) -> dict:
        text = str(value or "").strip()
        if action == "ask_jev":
            self.jev_decision = self.decision_tool.classify(
                self.user_text,
                history=self.history,
                turn_id=turn_id,
            )
            return asdict(self.jev_decision)
        if action == "delegate_data":
            self.suppress_reflection = True
            # The model cannot replace the user's path with a path discovered
            # in an untrusted document or an earlier tool result.
            response = self.data_agent.handle(self.user_text, turn_id=turn_id)
            return {"specialist": "data_agent", "reply": response.reply, "result": response.result}
        if action == "delegate_code":
            if not self.specialists:
                return {"status": "unavailable", "specialist": "coding_agent"}
            self.suppress_reflection = True
            return self.specialists.delegate("coding_agent", self.user_text, turn_id=turn_id, cancel=self.cancel)
        if action == "recall_memory":
            retrieval_started = time.monotonic()
            self.retrieved_memories = self.memory.recall(text or self.user_text, limit=8)
            self._log_memory_retrieval(
                mode="keyword",
                query=text or self.user_text,
                started=retrieval_started,
                turn_id=turn_id,
            )
            return {
                "memories": [asdict(item) for item in self.retrieved_memories],
                "count": len(self.retrieved_memories),
            }
        if action == "list_memories":
            retrieval_started = time.monotonic()
            self.retrieved_memories = self.memory.list(limit=20)
            self._log_memory_retrieval(
                mode="recent",
                query=None,
                started=retrieval_started,
                turn_id=turn_id,
            )
            return {
                "memories": [asdict(item) for item in self.retrieved_memories],
                "count": len(self.retrieved_memories),
            }
        if action == "save_memory":
            if not text:
                raise ValueError("Memory content is required")
            saved = self.memory.remember(text, source=f"explicit:turn:{turn_id}")
            self.suppress_reflection = True
            self.audit.emit(
                "memory.created",
                turn_id=turn_id,
                memory_id=saved.id,
                kind=saved.kind,
                content=saved.content,
                source="explicit",
                status="success",
            )
            return {"status": "saved", "memory": asdict(saved)}
        if action == "forget_memory":
            return self._forget_memory(text, turn_id=turn_id)
        if action == "search_knowledge":
            self.knowledge_hits = self.knowledge.search(
                text or self.user_text,
                limit=4,
                turn_id=turn_id,
            )
            return {
                "hits": [asdict(item) for item in self.knowledge_hits],
                "count": len(self.knowledge_hits),
            }
        if action == "list_sources":
            return {"sources": self.knowledge.list_documents()}
        if action == "runtime_status":
            return {
                "conversation_model": model_name(),
                "decision_model": self.decision_tool.model,
                "embedding_model": (
                    self.knowledge.embeddings.model if self.knowledge.embeddings else None
                ),
            }
        if action == "find_skill":
            return {"skills": self.skills.discover(text)}
        if action == "read_skill":
            result = self.skills.read(text)
            self.audit.emit("skill.loaded", turn_id=turn_id, name=text,
                            revision=result["revision"], status="success")
            return result
        if action == "end_session":
            self.end_session_requested = True
            self.suppress_reflection = True
            return {"status": "session_will_end_after_reply"}
        raise ValueError(f"Unknown Marcus action: {action}")

    def _forget_memory(self, text: str, *, turn_id: str | None) -> dict:
        if not text:
            raise ValueError("A memory id or query is required")
        target: Memory | None = None
        try:
            target = self.memory.get(text)
        except ValueError:
            matches = self.memory.recall(text, limit=5)
            if len(matches) != 1:
                return {
                    "status": "needs_choice",
                    "matches": [asdict(item) for item in matches],
                }
            target = matches[0]
        if target is None:
            return {"status": "not_found"}
        archived = self.memory.forget(target.id)
        self.suppress_reflection = True
        self.audit.emit(
            "memory.archived" if archived else "memory.archive.failed",
            turn_id=turn_id,
            memory_id=target.id,
            content=target.content,
            status="success" if archived else "not_found",
        )
        return {"status": "archived" if archived else "not_found", "memory": asdict(target)}

    def _log_memory_retrieval(
        self,
        *,
        mode: str,
        query: str | None,
        started: float,
        turn_id: str | None,
    ) -> None:
        self.audit.emit(
            "memory.retrieval.completed",
            turn_id=turn_id,
            query=query,
            memory_ids=[item.id for item in self.retrieved_memories],
            memories=[asdict(item) for item in self.retrieved_memories],
            result_count=len(self.retrieved_memories),
            mode=mode,
            duration_ms=round((time.monotonic() - started) * 1000),
            status="success",
        )
