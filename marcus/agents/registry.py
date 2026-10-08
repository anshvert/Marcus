from __future__ import annotations

from typing import Any

from marcus.observability.audit import AuditLog


class SpecialistRegistry:
    """The manager can delegate only to explicitly registered specialists."""

    def __init__(self, *, audit: AuditLog) -> None:
        self.audit = audit
        self._agents: dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        if name in self._agents:
            raise ValueError("Specialist is already registered")
        self._agents[name] = agent

    def has(self, name: str) -> bool:
        return name in self._agents

    def delegate(self, name: str, request: str, *, turn_id: str | None = None, cancel=None) -> dict:
        agent = self._agents.get(name)
        if agent is None:
            return {"status": "unavailable", "specialist": name}
        self.audit.emit("agent.delegated", turn_id=turn_id, from_agent="marcus", to_agent=name,
                        task=request[:240], status="delegated")
        kwargs = {"turn_id": turn_id}
        if name == "coding_agent":
            kwargs["cancel"] = cancel
        response = agent.handle(request, **kwargs)
        result = dict(response.result)
        result.pop("reply", None)
        return {"specialist": name, "reply": response.reply, "result": result}
