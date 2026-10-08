from __future__ import annotations

import json
import os
import secrets
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from marcus.observability.web import TraceDashboardHandler
from marcus.runtime.service import MarcusRuntime


def gateway_token(path: Path) -> str:
    configured = os.environ.get("MARCUS_GATEWAY_TOKEN")
    if configured:
        return configured
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        token = secrets.token_urlsafe(32)
        stream.write(token)
        return token


class GatewayHandler(TraceDashboardHandler):
    runtime: MarcusRuntime
    token: str

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        if not secrets.compare_digest(supplied, "Bearer " + self.token):
            self.send_error(401, "A gateway bearer token is required")
            return False
        return True

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            if not self._local_request() or not self._authorized():
                return
            if path == "/api/sessions":
                self._send_json({"sessions": self.runtime.state.sessions()})
            elif path == "/api/jobs":
                self._send_json({"jobs": self.runtime.state.jobs()})
            elif path.startswith("/api/jobs/"):
                result = self.runtime.state.job(path.rsplit("/", 1)[-1])
                if result is None:
                    self.send_error(404)
                else:
                    self._send_json(result)
            else:
                self.send_error(404)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if not self._local_request() or not self._authorized():
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 65536 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_error(400, "Send a JSON body of at most 64 KiB")
                return
            self.connection.settimeout(10)
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            path = urlparse(self.path).path
            if path == "/api/turns":
                result = self.runtime.submit(data.get("text"), session_id=data.get("session_id", "default"))
                self._send_json(result)
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                self._send_json({"cancel_requested": self.runtime.cancel(job_id)})
            elif path == "/api/speech/interrupt":
                if self.runtime.speech:
                    self.runtime.speech.interrupt()
                self._send_json({"status": "interrupted"})
            else:
                self.send_error(404)
        except (ValueError, TypeError) as exc:
            self.send_error(400, str(exc))
        except RuntimeError as exc:
            self.send_error(503, str(exc))


def make_gateway(runtime: MarcusRuntime, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The gateway supports localhost binding only")
    token = gateway_token(runtime.state.path.parent / "gateway-token")
    if len(token) < 16:
        raise ValueError("Gateway token must contain at least 16 characters")
    handler = type("ConfiguredGatewayHandler", (GatewayHandler,), {
        "runtime": runtime, "token": token, "bus": runtime.audit.bus,
        "log_root": runtime.audit.root, "stopping": threading.Event(),
    })
    server = ThreadingHTTPServer((host, port), handler)
    return server


def serve_gateway(runtime: MarcusRuntime, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    runtime.claim()
    server = make_gateway(runtime, host=host, port=port)
    url = f"http://{host}:{server.server_port}"
    runtime.audit.emit("gateway.started", endpoint=url, status="ready")
    print(f"Marcus gateway + trace: {url}\nConnect a terminal with: marcus --connect {url}")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print()
    finally:
        server.RequestHandlerClass.stopping.set()
        server.server_close()
        runtime.close()
