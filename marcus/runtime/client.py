from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class GatewayClient:
    def __init__(self, url: str, *, token_path: Path = Path("data/gateway-token")) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path not in {"", "/"}:
            raise ValueError("Connect to a localhost HTTP gateway URL")
        self.url = url.rstrip("/")
        self.token = os.environ.get("MARCUS_GATEWAY_TOKEN") or token_path.read_text(encoding="utf-8").strip()

    def request(self, path: str, data: dict | None = None) -> dict:
        request = Request(self.url + path, method="POST" if data is not None else "GET",
                          data=json.dumps(data).encode() if data is not None else None,
                          headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    def turn(self, text: str, *, session_id: str, on_text: Callable[[str], None]) -> dict:
        job = self.request("/api/turns", {"text": text, "session_id": session_id})
        job_id = job["id"]
        query = urlencode({"chat": "1", "session": session_id, "turn": job_id, "limit": "1000"})
        visible = False
        try:
            with urlopen(self.url + "/events?" + query, timeout=40) as response:
                for raw in response:
                    if not raw.startswith(b"data:"):
                        continue
                    event = json.loads(raw[5:])
                    if event["event"] == "chat.delta":
                        visible = True
                        on_text(event["data"]["text"])
                    if event["event"] in {"job.completed", "job.failed", "job.cancelled"}:
                        break
        except KeyboardInterrupt:
            self.request(f"/api/jobs/{job_id}/cancel", {})
            raise
        result = self.request(f"/api/jobs/{job_id}")
        if result["status"] != "completed":
            raise RuntimeError(result.get("error") or result["status"])
        if not visible:
            on_text(result["result"]["reply"])
        return result["result"]
