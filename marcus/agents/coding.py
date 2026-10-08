from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from marcus.agents.data import DataAgentResponse
from marcus.observability.audit import AuditLog


class CodexAgent:
    """Optional CLI specialist; fixed workspace, sandbox, and a bounded run."""

    name = "coding_agent"

    def __init__(self, workspace: Path, *, audit: AuditLog, command: str | None = None, timeout: float = 180) -> None:
        self.workspace = workspace.expanduser().resolve()
        if not self.workspace.is_dir() or self.workspace == Path(self.workspace.anchor) or self.workspace == Path.home():
            raise ValueError("Choose an existing project directory for the coding agent")
        self.audit, self.command, self.timeout = audit, command or shutil.which("codex"), timeout

    def handle(self, request: str, *, turn_id: str | None = None, cancel: threading.Event | None = None) -> DataAgentResponse:
        if not self.command:
            return DataAgentResponse("Codex CLI is not installed or on PATH.", {"status": "unavailable"})
        started = time.monotonic()
        self.audit.emit("coding_agent.started", turn_id=turn_id, workspace=str(self.workspace), status="running")
        arguments = [self.command, "exec", "--ignore-user-config", "--sandbox", "workspace-write",
                     "--config", 'approval_policy="never"', "--config", "sandbox_workspace_write.network_access=false",
                     "--json", "--color", "never", "--ephemeral", "--skip-git-repo-check", "--cd", str(self.workspace), "-"]
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"OPENROUTER_API_KEY", "MARCUS_CHAT_API_KEY", "MARCUS_GATEWAY_TOKEN"}}
        events: queue.Queue = queue.Queue(maxsize=256)
        process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, env=environment, start_new_session=True)
        reply = ""

        def read_events():
            for line in process.stdout:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Reasoning payloads are deliberately excluded from application
                # traces. Trace actions, outcomes, and public assistant output.
                item = record.get("item") or {}
                if item.get("type") == "reasoning":
                    continue
                if events.full():
                    try:
                        events.get_nowait()
                    except queue.Empty:
                        pass
                events.put_nowait(record)

        reader = threading.Thread(target=read_events, daemon=True)
        reader.start()
        try:
            process.stdin.write(request)
            process.stdin.close()
            while process.poll() is None or reader.is_alive() or not events.empty():
                if cancel and cancel.is_set():
                    raise InterruptedError("Coding task cancelled; existing edits are preserved")
                if time.monotonic() - started > self.timeout:
                    raise TimeoutError("Coding task exceeded its time budget; existing edits are preserved")
                try:
                    event = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                item = event.get("item") or {}
                if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                    reply = str(item.get("text", ""))[:12000]
                elif event.get("type") in {"item.started", "item.completed", "turn.failed", "error"}:
                    self.audit.emit("coding_agent.progress", turn_id=turn_id, event_type=event.get("type"),
                                    item_type=item.get("type"), command=str(item.get("command", ""))[:2000],
                                    status=item.get("status", "running"))
            if process.returncode != 0:
                raise RuntimeError("Codex CLI failed; inspect the coding-agent trace and CLI authentication")
            if not reply:
                raise RuntimeError("Codex CLI exited without a final assistant message")
            result = {"status": "completed", "workspace": str(self.workspace), "reply": reply}
            self.audit.emit("coding_agent.completed", turn_id=turn_id, duration_ms=round((time.monotonic() - started) * 1000), status="success")
            return DataAgentResponse(reply, result)
        except Exception as exc:
            self.audit.emit("coding_agent.failed", turn_id=turn_id, duration_ms=round((time.monotonic() - started) * 1000), summary=str(exc), status="failed")
            return DataAgentResponse(str(exc), {"status": "cancelled" if isinstance(exc, InterruptedError) else "failed", "error": str(exc)})
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            reader.join(timeout=2)
            if process.stdout:
                process.stdout.close()

