from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from marcus.core.config import load_dotenv
from marcus.core.doctor import diagnostics
from marcus.observability.audit import follow_events, format_event, latest_audit_path, read_events, session_scope
from marcus.observability.web import serve_trace_dashboard
from marcus.runtime.client import GatewayClient
from marcus.runtime.gateway import serve_gateway
from marcus.runtime.service import build_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marcus personal agent")
    parser.add_argument("--check", action="store_true", help="Test the configured conversation model and exit")
    parser.add_argument("--session", default="default", help="Named conversation; resumes saved history")
    parser.add_argument("--connect", metavar="URL", help="Connect this terminal to an existing local gateway")
    commands = parser.add_subparsers(dest="command")
    trace = commands.add_parser("trace", help="View the separate audit stream")
    trace.add_argument("--follow", action="store_true")
    trace.add_argument("--payloads", action="store_true", help="Show full prompt and retrieval payloads")
    trace.add_argument("--verbose", action="store_true", help="Print complete JSON events")
    trace.add_argument("--limit", type=int, default=30)
    trace.add_argument("--web", action="store_true")
    trace.add_argument("--host", default="127.0.0.1")
    trace.add_argument("--port", type=int, default=8765)
    trace.add_argument("--no-open", action="store_true")
    serve = commands.add_parser("serve", help="Run the persistent local gateway and trace dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true")
    commands.add_parser("doctor", help="Check configuration locally without model calls")
    commands.add_parser("sessions", help="List persisted conversation sessions")
    commands.add_parser("jobs", help="List recent gateway jobs")
    ingest = commands.add_parser("ingest", help="Ingest a PDF or Markdown file")
    ingest.add_argument("path")
    ingest.add_argument("--label")
    ingest.add_argument("--type", dest="document_type", default="document", choices=["resume", "profile", "notes", "reference", "document"])
    commands.add_parser("sources", help="List ingested documents")
    return parser


def _trace(*, follow: bool, payloads: bool, verbose: bool, limit: int) -> None:
    root = Path(os.environ.get("MARCUS_LOG_DIR", "data/logs"))
    path = latest_audit_path(root)
    if path is None:
        if not follow:
            print("No Marcus audit log exists yet.")
            return
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"audit-{datetime.now(timezone.utc).date().isoformat()}.jsonl"
        path.touch()

    def display(record):
        print(json.dumps(record, indent=2, ensure_ascii=False) if verbose else format_event(record, include_payloads=payloads))

    for record in read_events(path, limit=limit):
        display(record)
    if follow:
        print(f"Following {path}; press Ctrl-C to stop.")
        try:
            for record in follow_events(path):
                display(record)
        except KeyboardInterrupt:
            print()


def _conversation(turn, *, session_id: str, speech=None) -> None:
    print(f"Marcus: Ready. Session: {session_id}. Type what you need; Ctrl-C exits.")
    while True:
        try:
            text = input("You: ").strip()
            if not text:
                continue
            print("Marcus: ", end="", flush=True)
            result = turn(text, session_id=session_id, on_text=lambda delta: print(delta, end="", flush=True))
            print()
            if result.get("end_session"):
                if speech:
                    speech.drain()
                return
        except (EOFError, KeyboardInterrupt):
            print("\nMarcus: Until next time.")
            return
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"\nMarcus: {exc}")


def main() -> None:
    args = _parser().parse_args()
    load_dotenv()
    if args.command == "doctor":
        for item in diagnostics():
            print(f"{'OK' if item['ok'] else 'CHECK'}  {item['check']}: {item['detail']}")
        return
    if args.command == "trace":
        if args.web:
            serve_trace_dashboard(host=args.host, port=args.port, log_root=Path(os.environ.get("MARCUS_LOG_DIR", "data/logs")), open_browser=not args.no_open)
        else:
            _trace(follow=args.follow, payloads=args.payloads, verbose=args.verbose, limit=args.limit)
        return
    if args.connect:
        client = GatewayClient(args.connect, token_path=Path(os.environ.get("MARCUS_RUNTIME_DB", "data/runtime.sqlite")).parent / "gateway-token")
        if args.command in {"jobs", "sessions"}:
            print(json.dumps(client.request("/api/" + args.command), indent=2))
        elif args.command or args.check:
            raise SystemExit("Use natural language in the connected session; admin subcommands operate locally.")
        else:
            _conversation(client.turn, session_id=args.session)
        return

    runtime = build_runtime(voice=args.command == "serve" or (args.command is None and not args.check))
    try:
        if args.command == "serve":
            serve_gateway(runtime, host=args.host, port=args.port, open_browser=not args.no_open)
            return
        if args.command == "sessions":
            print(json.dumps(runtime.state.sessions(), indent=2))
            return
        if args.command == "jobs":
            print(json.dumps(runtime.state.jobs(), indent=2))
            return
        if args.command == "sources":
            print(json.dumps(runtime.knowledge.list_documents(), indent=2))
            return
        if args.command == "ingest":
            result = runtime.data_agent.ingest_path(args.path, label=args.label, document_type=args.document_type)
            if result.get("status") == "failed":
                raise SystemExit(f"Ingestion failed: {result['error']}")
            print(json.dumps(result, indent=2))
            return
        if args.check:
            if runtime.brain is None:
                raise SystemExit("No conversation model is configured; run marcus doctor.")
            response = runtime.brain.respond("Reply briefly that Marcus is online. Do not create a memory.", history=[], memories=[])
            print(f"Marcus model check ({runtime.brain.model}): {response.reply}")
            return
        runtime.claim()
        if runtime.brain is None:
            raise SystemExit("No conversation model is configured; run marcus doctor.")
        with session_scope(args.session):
            runtime.audit.emit("session.started", model=runtime.brain.model, status="active")

        def local_turn(text, **options):
            from dataclasses import asdict
            return asdict(runtime.turn(text, **options))

        _conversation(local_turn, session_id=args.session, speech=runtime.speech)
        with session_scope(args.session):
            runtime.audit.emit("session.ended", status="completed")
    finally:
        runtime.close()
