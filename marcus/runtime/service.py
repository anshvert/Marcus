from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from marcus.agents.curator import BackgroundReflector, ReflectionClient
from marcus.agents.data import DataAgent
from marcus.agents.coding import CodexAgent
from marcus.agents.registry import SpecialistRegistry
from marcus.agents.manager import BrainError, OpenRouterBrain, TurnCancelled
from marcus.core.config import load_dotenv
from marcus.core.context import bounded_history
from marcus.core.policy import CapabilityPolicy
from marcus.knowledge.embeddings import EmbeddingError, OpenRouterEmbeddings
from marcus.knowledge.ingest import DocumentIngestor
from marcus.knowledge.store import KnowledgeStore
from marcus.memory.curation import MemoryCurator
from marcus.memory.store import MarkdownMemoryStore
from marcus.memory.style import StyleProfileStore
from marcus.observability.audit import AuditLog, session_scope
from marcus.providers.base import Brain
from marcus.runtime.state import RuntimeStore
from marcus.skills.store import SkillStore
from marcus.tools.jev import JevDecisionTool
from marcus.tools.manager import MarcusToolbox
from marcus.voice.tts import SpeechOutput


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    session_id: str
    reply: str
    end_session: bool
    duration_ms: int


class MarcusRuntime:
    """One serialized foreground lane; curator and audio use background lanes."""

    def __init__(self, *, audit: AuditLog, memory: MarkdownMemoryStore,
                 knowledge: KnowledgeStore, data_agent: DataAgent, brain: Brain | None,
                 style: StyleProfileStore, reflector: BackgroundReflector | None,
                 decision_tool: JevDecisionTool, state: RuntimeStore,
                 speech: SpeechOutput | None = None, skills: SkillStore | None = None,
                 policy: CapabilityPolicy | None = None) -> None:
        self.audit, self.memory, self.knowledge = audit, memory, knowledge
        self.data_agent, self.brain, self.style = data_agent, brain, style
        self.reflector, self.decision_tool, self.state = reflector, decision_tool, state
        self.speech, self.skills = speech, skills or SkillStore()
        self.policy = policy or CapabilityPolicy()
        self.specialists = SpecialistRegistry(audit=audit)
        self.specialists.register("data_agent", data_agent)
        self._lane = threading.Lock()
        self._jobs_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="marcus-turn")
        self._cancels: dict[str, threading.Event] = {}
        self._active_job: str | None = None
        self._closed = False
        self._lease = None

    def claim(self) -> None:
        """Prevent two chat runtimes from sharing jobs/audio accidentally."""
        if self._lease is not None:
            return
        import fcntl
        lease = (self.state.path.parent / "runtime.lock").open("a+")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            raise RuntimeError("Marcus is already running for this data directory. Connect using --connect.")
        self._lease = lease
        count = self.state.recover()
        self.audit.emit("runtime.ready", interrupted_jobs=count, status="ready")

    def turn(self, text: str, *, session_id: str = "default", turn_id: str | None = None,
             cancel: threading.Event | None = None, on_text: Callable[[str], None] | None = None) -> TurnResult:
        if self._closed:
            raise RuntimeError("Runtime is shutting down")
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError("A turn needs 1–16000 characters of text")
        self.state.ensure_session(session_id)
        turn_id = turn_id or uuid.uuid4().hex[:12]
        cancel = cancel or threading.Event()
        with self._lane, session_scope(session_id):
            started = time.monotonic()
            if cancel.is_set():
                raise TurnCancelled("Turn cancelled before starting")
            if self.brain is None:
                raise BrainError("No conversation model is configured")
            if self.speech:
                self.speech.interrupt(turn_id=turn_id)
            self.audit.emit("turn.received", turn_id=turn_id, input=text, status="running")
            history = bounded_history(self.state.history(session_id))
            toolbox = MarcusToolbox(decision_tool=self.decision_tool, data_agent=self.data_agent,
                                    memory=self.memory, knowledge=self.knowledge, audit=self.audit,
                                    skills=self.skills, policy=self.policy, cancel=cancel, specialists=self.specialists)
            toolbox.begin_turn(text, history)
            output: list[str] = []

            def present(delta: str) -> None:
                if cancel.is_set():
                    raise TurnCancelled("Turn cancelled")
                output.append(delta)
                # Text deltas are live-only. Persist the complete reply once,
                # rather than generating a JSONL record for each token.
                self.audit.bus.publish({"schema_version": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "event_id": uuid.uuid4().hex[:12], "event": "chat.delta",
                                        "session_id": session_id, "turn_id": turn_id,
                                        "data": {"text": delta, "status": "streaming"}})
                if on_text:
                    on_text(delta)
                if self.speech:
                    self.speech.feed(delta, turn_id=turn_id)

            try:
                core_context = self.memory.core_context()
                if self.memory.sync_errors:
                    self.audit.emit("memory.index.warning", turn_id=turn_id,
                                    excluded_notes=self.memory.sync_errors, status="warning")
                response = self.brain.respond(text, history=history, memories=[],
                                              source_catalog=toolbox.source_catalog(),
                                              style_context=self.style.prompt_context(),
                                              core_context=core_context,
                                              skill_catalog=self.skills.catalog(),
                                              tools=toolbox.schemas, tool_executor=toolbox.execute,
                                              on_text=present, turn_id=turn_id, cancel=cancel)
                if cancel.is_set():
                    raise TurnCancelled("Turn cancelled after generation")
                if not output:
                    present(response.reply)
                if self.speech:
                    self.speech.finish(turn_id=turn_id)
                self.state.save_turn(session_id, turn_id, text, response.reply)
                history = bounded_history(self.state.history(session_id))
                if self.reflector and response.reflect and not toolbox.suppress_reflection:
                    state = {"latest_turn": {"user": text, "assistant": response.reply},
                             "recent_history": history,
                             "retrieved_memories": [asdict(item) for item in toolbox.retrieved_memories],
                             "knowledge_references": [{"source": item.title, "heading": item.heading,
                                                       "page": item.page, "chunk_id": item.chunk_id}
                                                      for item in toolbox.knowledge_hits],
                             "current_style": asdict(self.style.load())}
                    self.reflector.submit(state, active_memory_ids={item.id for item in toolbox.retrieved_memories}, turn_id=turn_id)
                else:
                    self.audit.emit("curator.skipped", turn_id=turn_id,
                                    summary="persistence already handled or no durable signal", status="skipped")
                duration = round((time.monotonic() - started) * 1000)
                self.audit.emit("turn.completed", turn_id=turn_id, duration_ms=duration, phase="foreground", status="success")
                return TurnResult(turn_id, session_id, response.reply, toolbox.end_session_requested, duration)
            except (Exception, KeyboardInterrupt) as exc:
                if self.speech:
                    self.speech.interrupt(turn_id=turn_id)
                cancelled = isinstance(exc, (TurnCancelled, KeyboardInterrupt))
                self.audit.emit("turn.cancelled" if cancelled else "turn.failed",
                                turn_id=turn_id, duration_ms=round((time.monotonic() - started) * 1000),
                                summary=str(exc)[:240], status="cancelled" if cancelled else "failed")
                raise

    def submit(self, text: str, *, session_id: str = "default") -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError("A turn needs 1–16000 characters of text")
        with self._jobs_lock:
            if self._closed:
                raise RuntimeError("Runtime is shutting down")
            if len(self._cancels) >= 32:
                raise RuntimeError("Turn queue is full")
            job_id = self.state.create_job(session_id)
            cancel = threading.Event()
            self._cancels[job_id] = cancel
            with session_scope(session_id):
                self.audit.emit("job.queued", turn_id=job_id, job_id=job_id, status="queued")
            self._executor.submit(self._run_job, job_id, session_id, text, cancel, time.monotonic())
        return self.state.job(job_id)

    def _run_job(self, job_id: str, session_id: str, text: str, cancel: threading.Event, queued: float) -> None:
        with session_scope(session_id):
            try:
                with self._jobs_lock:
                    if cancel.is_set():
                        raise TurnCancelled("Cancelled while queued")
                    self._active_job = job_id
                    self.state.update_job(job_id, "running")
                self.audit.emit("job.started", turn_id=job_id, queue_wait_ms=round((time.monotonic() - queued) * 1000), status="running")
                result = self.turn(text, session_id=session_id, turn_id=job_id, cancel=cancel)
                self.state.update_job(job_id, "completed", result=asdict(result))
                self.audit.emit("job.completed", turn_id=job_id, status="success")
            except Exception as exc:
                status = "cancelled" if isinstance(exc, TurnCancelled) else "failed"
                self.state.update_job(job_id, status, error=str(exc)[:500])
                self.audit.emit(f"job.{status}", turn_id=job_id, summary=str(exc)[:240], status=status)
            finally:
                with self._jobs_lock:
                    self._cancels.pop(job_id, None)
                    if self._active_job == job_id:
                        self._active_job = None

    def cancel(self, job_id: str) -> bool:
        with self._jobs_lock:
            event = self._cancels.get(job_id)
            if event is None:
                return False
            event.set()
            if self._active_job == job_id and self.speech:
                self.speech.interrupt(turn_id=job_id)
            job = self.state.job(job_id)
            with session_scope(job["session_id"]):
                self.audit.emit("job.cancel_requested", turn_id=job_id, status="cancelling")
            return True

    def close(self) -> None:
        with self._jobs_lock:
            if self._closed:
                return
            self._closed = True
            for event in self._cancels.values():
                event.set()
        if self.speech:
            self.speech.close()
        self._executor.shutdown(wait=True)
        if self.reflector:
            self.reflector.close()
        if self._lease:
            self._lease.close()
            self._lease = None


def build_runtime(*, voice: bool = True) -> MarcusRuntime:
    load_dotenv()
    audit = AuditLog(Path(os.environ.get("MARCUS_LOG_DIR", "data/logs")))
    memory = MarkdownMemoryStore()
    style = StyleProfileStore(memory.vault_path)
    try:
        embeddings = OpenRouterEmbeddings(audit=audit) if os.environ.get("MARCUS_EMBEDDINGS", "1") != "0" else None
    except EmbeddingError:
        embeddings = None
    knowledge = KnowledgeStore(Path(os.environ.get("MARCUS_KNOWLEDGE_DB", "data/knowledge.sqlite")), embeddings=embeddings, audit=audit)
    ingestor = DocumentIngestor(knowledge, vault_path=memory.vault_path, embeddings=embeddings, audit=audit)
    data_agent = DataAgent(ingestor, audit=audit)
    decision_tool = JevDecisionTool(audit=audit)
    try:
        brain = OpenRouterBrain(audit=audit)
    except BrainError:
        brain = None
    reflector = None
    if brain is not None and os.environ.get("MARCUS_REFLECTION", "1") != "0":
        reflector = BackgroundReflector(ReflectionClient(audit=audit), MemoryCurator(memory), style, audit=audit)
    denied = set(filter(None, (item.strip() for item in os.environ.get("MARCUS_DENY_ACTIONS", "").split(","))))
    policy = CapabilityPolicy()
    policy.allowed -= denied
    runtime = MarcusRuntime(audit=audit, memory=memory, knowledge=knowledge, data_agent=data_agent,
                         brain=brain, style=style, reflector=reflector, decision_tool=decision_tool,
                         state=RuntimeStore(Path(os.environ.get("MARCUS_RUNTIME_DB", "data/runtime.sqlite"))),
                         skills=SkillStore(Path(os.environ.get("MARCUS_SKILL_DIR", "skills"))), policy=policy,
                         speech=SpeechOutput(audit=audit) if voice else None)
    if os.environ.get("MARCUS_CODEX_ENABLED", "0").casefold() in {"1", "true", "yes"}:
        workspace = os.environ.get("MARCUS_CODEX_WORKSPACE")
        if not workspace:
            runtime.close()
            raise ValueError("Set MARCUS_CODEX_WORKSPACE before enabling the coding specialist")
        runtime.specialists.register("coding_agent", CodexAgent(Path(workspace), audit=audit,
                                    timeout=float(os.environ.get("MARCUS_CODEX_TIMEOUT", "180"))))
    return runtime
