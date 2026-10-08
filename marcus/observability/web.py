from __future__ import annotations

import json
import threading
import time
import webbrowser
import queue
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from marcus.observability.audit import latest_audit_path, read_events


class TraceDashboardHandler(BaseHTTPRequestHandler):
    log_root = Path("data/logs")
    dashboard_path = Path(__file__).with_name("dashboard.html")
    bus = None
    stopping = None

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in expected or (origin and origin != "http://" + host) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.send_error(403, "Only same-origin localhost requests are supported")
            return False
        return True

    def do_GET(self) -> None:
        if not self._local_request():
            return
        request = urlparse(self.path)
        if request.path == "/":
            self._serve_dashboard()
            return
        if request.path == "/events":
            values = parse_qs(request.query)
            try:
                limit = min(1_000, max(1, int(values.get("limit", ["250"])[0])))
            except ValueError:
                limit = 250
            self._serve_events(limit)
            return
        if request.path == "/health":
            self._send_json({"status": "ok"})
            return
        self.send_error(404)

    def _serve_dashboard(self) -> None:
        body = self.dashboard_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self, limit: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        if self.bus is not None:
            self._serve_bus_events(limit)
            return

        path = self._current_log()
        try:
            with path.open("r", encoding="utf-8") as stream:
                for record in read_events(path, limit=limit):
                    self._send_event(json.dumps(record))
                stream.seek(0, 2)
                pending = ""
                heartbeat = time.monotonic()
                while True:
                    line = stream.readline()
                    if line:
                        pending += line
                        if pending.endswith("\n"):
                            self._send_event(pending)
                            pending = ""
                        continue
                    newest = latest_audit_path(self.log_root)
                    if newest is not None and newest != path:
                        break
                    if time.monotonic() - heartbeat > 10:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        heartbeat = time.monotonic()
                    time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_bus_events(self, limit: int) -> None:
        values = parse_qs(urlparse(self.path).query)
        session = values.get("session", [None])[0]
        turn = values.get("turn", [None])[0]
        include_chat = values.get("chat", ["0"])[0] == "1"
        seen: set[str] = set()

        def send(record):
            event_id = record.get("event_id")
            if event_id in seen or (session and record.get("session_id") != session) or (turn and record.get("turn_id") != turn):
                return
            if record.get("event") == "chat.delta" and not include_chat:
                return
            seen.add(event_id)
            if len(seen) > 2000:
                seen.clear()
                seen.add(event_id)
            self._send_event(json.dumps(record))

        try:
            # Subscribe before reading disk so a concurrent event cannot fall
            # into the gap between replay and live delivery.
            with self.bus.subscribe(replay=limit) as events:
                path = latest_audit_path(self.log_root)
                if path:
                    for record in read_events(path, limit=limit):
                        send(record)
                while self.stopping is None or not self.stopping.is_set():
                    try:
                        send(events.get(timeout=5))
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _current_log(self) -> Path:
        path = latest_audit_path(self.log_root)
        if path is not None:
            return path
        self.log_root.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).date().isoformat()
        path = self.log_root / f"audit-{day}.jsonl"
        path.touch()
        return path

    def _send_event(self, line: str) -> None:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        self.wfile.write(f"id: {record.get('event_id', '')}\ndata: {line.strip()}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _send_json(self, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve_trace_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    log_root: Path = Path("data/logs"),
    open_browser: bool = True,
) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Trace server supports localhost binding only")
    handler = type(
        "ConfiguredTraceDashboardHandler",
        (TraceDashboardHandler,),
        {"log_root": log_root},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}"
    print(f"Marcus trace dashboard: {url}")
    print("Press Ctrl-C to stop it.")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
