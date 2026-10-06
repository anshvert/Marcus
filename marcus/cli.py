from __future__ import annotations

import argparse
import json
import os
import shlex
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from marcus.brain import BrainError, OpenRouterBrain
from marcus.config import load_dotenv, model_name
from marcus.curator import MemoryCurator
from marcus.data_agent import DataAgent
from marcus.embeddings import EmbeddingError, OpenRouterEmbeddings
from marcus.events import AuditLog, follow_events, format_event, latest_audit_path, read_events
from marcus.ingest import DocumentIngestor
from marcus.jev import JevRouter
from marcus.knowledge import KnowledgeStore
from marcus.memory import MarkdownMemoryStore
from marcus.reflection import BackgroundReflector, ReflectionClient
from marcus.style import StyleProfileStore


HELP = """Commands:
  /remember <text>      Save an explicit memory
  /recall <query>       Search active memories
  /memories             List recent memories
  /forget <memory-id>   Move a memory to the archive
  /ingest <path>        Ingest a PDF or Markdown file
  /sources              List ingested source documents
  /model                Show the configured model
  /trace                Show how to open the separate live trace
  /help                 Show this help
  /quit                 Exit Marcus
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marcus personal agent")
    parser.add_argument("--check", action="store_true", help="Test the configured model and exit")
    commands = parser.add_subparsers(dest="command")

    trace = commands.add_parser("trace", help="View the audit event stream")
    trace.add_argument("--follow", action="store_true", help="Continue printing new events")
    trace.add_argument(
        "--payloads",
        action="store_true",
        help="Show complete prompts, retrieved chunks, embeddings, and JEV answers",
    )
    trace.add_argument("--verbose", action="store_true", help="Print complete JSON events")
    trace.add_argument("--limit", type=int, default=30, help="Number of existing events to show")

    ingest = commands.add_parser("ingest", help="Ingest a PDF or Markdown file")
    ingest.add_argument("path")
    ingest.add_argument("--label")
    ingest.add_argument(
        "--type",
        dest="document_type",
        default="document",
        choices=["resume", "profile", "notes", "reference", "document"],
    )

    commands.add_parser("sources", help="List ingested source documents")
    return parser


def _show_memories(memories: list) -> None:
    if not memories:
        print("Marcus: I found no matching memories.")
        return
    for memory in memories:
        print(f"- {memory.id}: {memory.content}")


def _show_sources(knowledge: KnowledgeStore) -> None:
    documents = knowledge.list_documents()
    if not documents:
        print("Marcus: No source documents have been ingested.")
        return
    for document in documents:
        print(
            f"- {document['id']}: {document['title']} "
            f"({document['document_type']}, {document['chunk_count']} chunks)"
        )


def _trace(*, follow: bool, payloads: bool, verbose: bool, limit: int) -> None:
    path = latest_audit_path()
    if path is None:
        if not follow:
            print("No Marcus audit log exists yet.")
            return
        root = Path("data/logs")
        root.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).date().isoformat()
        path = root / f"audit-{day}.jsonl"
        path.touch()

    def display(record: dict) -> None:
        print(
            json.dumps(record, indent=2, ensure_ascii=False)
            if verbose
            else format_event(record, include_payloads=payloads)
        )

    for record in read_events(path, limit=limit):
        display(record)
    if follow:
        print(f"-- following {path}; press Ctrl-C to stop --")
        try:
            for record in follow_events(path):
                display(record)
        except KeyboardInterrupt:
            print()


def _runtime():
    load_dotenv()
    audit = AuditLog()
    memory = MarkdownMemoryStore()
    curator = MemoryCurator(memory)
    style = StyleProfileStore(memory.vault_path)
    try:
        embeddings = OpenRouterEmbeddings(audit=audit)
    except EmbeddingError:
        embeddings = None
    knowledge = KnowledgeStore(
        Path(os.environ.get("MARCUS_KNOWLEDGE_DB", "data/knowledge.sqlite")),
        embeddings=embeddings,
        audit=audit,
    )
    ingestor = DocumentIngestor(
        knowledge,
        vault_path=memory.vault_path,
        embeddings=embeddings,
        audit=audit,
    )
    data_agent = DataAgent(ingestor, audit=audit)
    router = JevRouter(audit=audit)
    try:
        brain = OpenRouterBrain(audit=audit)
    except BrainError:
        brain = None
    reflector = None
    if brain is not None:
        reflector = BackgroundReflector(
            ReflectionClient(audit=audit),
            curator,
            style,
            audit=audit,
        )
    return audit, memory, knowledge, data_agent, router, brain, style, reflector


def _social_reply(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    if "thank" in normalized:
        return "Anytime."
    if normalized in {"bye", "goodbye", "see you", "later"}:
        return "See you."
    if "up" in normalized or normalized == "sup":
        return "Hey — I’m good. What’s up?"
    return "Hey. What can I help with?"


def main() -> None:
    args = _parser().parse_args()
    if args.command == "trace":
        _trace(
            follow=args.follow,
            payloads=args.payloads,
            verbose=args.verbose,
            limit=args.limit,
        )
        return

    audit, memory, knowledge, data_agent, router, brain, style, reflector = _runtime()

    if args.command == "sources":
        _show_sources(knowledge)
        return
    if args.command == "ingest":
        result = data_agent.ingest_path(
            args.path,
            label=args.label,
            document_type=args.document_type,
        )
        if result.get("status") == "failed":
            raise SystemExit(f"Ingestion failed: {result['error']}")
        print(json.dumps(result, indent=2))
        return

    if args.check:
        if brain is None:
            raise SystemExit("Marcus model check failed: OPENROUTER_API_KEY is not configured")
        try:
            response = brain.respond(
                "Reply briefly that Marcus is online. Do not create a memory.",
                history=[],
                memories=[],
            )
        except BrainError as exc:
            raise SystemExit(f"Marcus model check failed: {exc}") from exc
        print(f"Marcus model check ({model_name()}): {response.reply}")
        return

    history: list[dict[str, str]] = []
    audit.emit("session.started", model=model_name(), status="active")
    if brain:
        print(f"Marcus: Ready. Brain: {model_name()}. Memory: Markdown vault.")
    else:
        print("Marcus: Ready in memory-only mode; OPENROUTER_API_KEY is not configured.")
    print("Marcus: Type /help to see available commands.")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            if reflector:
                reflector.close()
            audit.emit("session.ended", status="completed")
            print("\nMarcus: Until next time.")
            return

        if not raw:
            continue
        turn_started = time.monotonic()
        turn_id = uuid.uuid4().hex[:12]
        audit.emit(
            "turn.received",
            turn_id=turn_id,
            input=raw,
            input_length=len(raw),
            status="running",
        )

        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            print(f"Marcus: I couldn't parse that command: {exc}")
            audit.emit("turn.failed", turn_id=turn_id, summary="command_parse_error", status="failed")
            continue

        command = parts[0].lower()
        argument = " ".join(parts[1:])

        try:
            if command == "/quit":
                if reflector:
                    audit.emit("curator.shutdown.started", turn_id=turn_id, status="waiting")
                    reflector.close()
                audit.emit("session.ended", turn_id=turn_id, status="completed")
                print("Marcus: Until next time.")
                return
            if command == "/help":
                print(HELP)
            elif command == "/remember":
                memory_write_started = time.monotonic()
                saved = memory.remember(argument)
                audit.emit(
                    "memory.created",
                    turn_id=turn_id,
                    memory_id=saved.id,
                    kind=saved.kind,
                    content=saved.content,
                    source="explicit",
                    duration_ms=round((time.monotonic() - memory_write_started) * 1000),
                    status="success",
                )
                print(f"Marcus: Remembered. Memory id: {saved.id}")
            elif command == "/recall":
                retrieval_started = time.monotonic()
                recalled = memory.recall(argument)
                audit.emit(
                    "memory.retrieval.completed",
                    turn_id=turn_id,
                    memory_ids=[item.id for item in recalled],
                    memories=[
                        {"id": item.id, "kind": item.kind, "content": item.content}
                        for item in recalled
                    ],
                    result_count=len(recalled),
                    mode="keyword",
                    duration_ms=round((time.monotonic() - retrieval_started) * 1000),
                    status="success",
                )
                _show_memories(recalled)
            elif command == "/memories":
                retrieval_started = time.monotonic()
                recalled = memory.list()
                audit.emit(
                    "memory.retrieval.completed",
                    turn_id=turn_id,
                    memory_ids=[item.id for item in recalled],
                    memories=[
                        {"id": item.id, "kind": item.kind, "content": item.content}
                        for item in recalled
                    ],
                    result_count=len(recalled),
                    mode="recent",
                    duration_ms=round((time.monotonic() - retrieval_started) * 1000),
                    status="success",
                )
                _show_memories(recalled)
            elif command == "/sources":
                _show_sources(knowledge)
            elif command == "/model":
                print(f"Marcus: {model_name()}")
            elif command == "/trace":
                print(
                    "Marcus: Run `.venv/bin/python -m marcus trace --follow` for the concise "
                    "trace, or add `--payloads` for complete prompts and chunks."
                )
            elif command == "/ingest":
                result = data_agent.ingest_path(argument, turn_id=turn_id)
                if result.get("status") == "failed":
                    print(f"Marcus: The data agent could not ingest that file: {result['error']}")
                else:
                    print(
                        f"Marcus: The data agent ingested {result['title']} into "
                        f"{result['chunk_count']} chunks ({result['embedding_status']})."
                    )
            elif command == "/forget":
                memory_write_started = time.monotonic()
                existing = memory.get(argument)
                forgotten = memory.forget(argument)
                audit.emit(
                    "memory.archived" if forgotten else "memory.archive.failed",
                    turn_id=turn_id,
                    memory_id=argument,
                    content=existing.content if existing else None,
                    duration_ms=round((time.monotonic() - memory_write_started) * 1000),
                    status="success" if forgotten else "not_found",
                )
                message = "Archived; it can be recovered." if forgotten else "I couldn't find that memory."
                print(f"Marcus: {message}")
            else:
                route = router.classify(raw, history=history, turn_id=turn_id)
                if route.intent == "social" and route.confidence >= 0.7:
                    reply = _social_reply(raw)
                    audit.emit(
                        "reply.fast_path",
                        turn_id=turn_id,
                        route="social",
                        reply=reply,
                        status="success",
                    )
                    print(f"Marcus: {reply}")
                    history.extend(
                        [{"role": "user", "content": raw}, {"role": "assistant", "content": reply}]
                    )
                    history = history[-12:]
                    audit.emit(
                        "curator.skipped",
                        turn_id=turn_id,
                        summary="router found no durable memory or style signal",
                        status="skipped",
                    )
                    audit.emit(
                        "turn.completed",
                        turn_id=turn_id,
                        duration_ms=round((time.monotonic() - turn_started) * 1000),
                        phase="foreground",
                        status="success",
                    )
                    continue

                if route.intent == "data_agent" and route.confidence >= 0.7:
                    response = data_agent.handle(raw, turn_id=turn_id)
                    print(f"Marcus: {response.reply}")
                    history.extend(
                        [
                            {"role": "user", "content": raw},
                            {"role": "assistant", "content": response.reply},
                        ]
                    )
                    history = history[-12:]
                    audit.emit(
                        "curator.skipped",
                        turn_id=turn_id,
                        summary="data-agent operations are not personal-memory candidates",
                        status="skipped",
                    )
                    audit.emit(
                        "turn.completed",
                        turn_id=turn_id,
                        duration_ms=round((time.monotonic() - turn_started) * 1000),
                        phase="foreground",
                        status="success",
                    )
                    continue

                if brain is None:
                    print(
                        "Marcus: I don't have a language model connected. "
                        "Configure OPENROUTER_API_KEY or use the local commands."
                    )
                    audit.emit(
                        "turn.completed",
                        turn_id=turn_id,
                        duration_ms=round((time.monotonic() - turn_started) * 1000),
                        phase="foreground",
                        summary="no generative model configured",
                        status="limited",
                    )
                    continue
                if route.needs_memory:
                    retrieval_started = time.monotonic()
                    relevant = memory.recall(raw, limit=8)
                    audit.emit(
                        "memory.retrieval.completed",
                        turn_id=turn_id,
                        memory_ids=[item.id for item in relevant],
                        memories=[
                            {"id": item.id, "kind": item.kind, "content": item.content}
                            for item in relevant
                        ],
                        result_count=len(relevant),
                        mode="keyword",
                        duration_ms=round((time.monotonic() - retrieval_started) * 1000),
                        status="success",
                    )
                else:
                    relevant = []
                    audit.emit(
                        "memory.retrieval.skipped",
                        turn_id=turn_id,
                        summary="JEV gate: memory is unnecessary",
                        duration_ms=0,
                        status="skipped",
                    )
                if route.needs_knowledge:
                    knowledge_hits = knowledge.search(raw, limit=4, turn_id=turn_id)
                else:
                    knowledge_hits = []
                    audit.emit(
                        "knowledge.retrieval.skipped",
                        turn_id=turn_id,
                        summary="JEV gate: document knowledge is unnecessary",
                        duration_ms=0,
                        status="skipped",
                    )
                response = brain.respond(
                    raw,
                    history=history,
                    memories=relevant,
                    knowledge_context=(knowledge.format_context(knowledge_hits) if knowledge_hits else ""),
                    style_context=style.prompt_context(),
                    turn_id=turn_id,
                )
                print(f"Marcus: {response.reply}")
                history.extend(
                    [
                        {"role": "user", "content": raw},
                        {"role": "assistant", "content": response.reply},
                    ]
                )
                history = history[-12:]
                if reflector and route.needs_reflection:
                    state_started = time.monotonic()
                    reflection_state = {
                        "latest_turn": {"user": raw, "assistant": response.reply},
                        "recent_history": list(history),
                        "retrieved_memories": [
                            {"id": item.id, "kind": item.kind, "content": item.content}
                            for item in relevant
                        ],
                        "knowledge_references": [
                            {
                                "source": item.title,
                                "heading": item.heading,
                                "page": item.page,
                                "chunk_id": item.chunk_id,
                            }
                            for item in knowledge_hits
                        ],
                        "current_style": asdict(style.load()),
                    }
                    audit.emit(
                        "curator.state.prepared",
                        turn_id=turn_id,
                        history_message_count=len(history),
                        retrieved_memory_count=len(relevant),
                        knowledge_hit_count=len(knowledge_hits),
                        duration_ms=round((time.monotonic() - state_started) * 1000),
                        status="success",
                    )
                    reflector.submit(
                        reflection_state,
                        active_memory_ids={item.id for item in relevant},
                        turn_id=turn_id,
                    )
                else:
                    audit.emit(
                        "curator.skipped",
                        turn_id=turn_id,
                        summary="JEV gate: no durable memory or style signal",
                        status="skipped",
                    )
            audit.emit(
                "turn.completed",
                turn_id=turn_id,
                duration_ms=round((time.monotonic() - turn_started) * 1000),
                phase="foreground",
                status="success",
            )
        except (ValueError, RuntimeError, BrainError) as exc:
            audit.emit(
                "turn.failed",
                turn_id=turn_id,
                error_type=type(exc).__name__,
                summary=str(exc)[:240],
                duration_ms=round((time.monotonic() - turn_started) * 1000),
                status="failed",
            )
            print(f"Marcus: {exc}")
