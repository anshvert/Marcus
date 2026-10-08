from __future__ import annotations

from typing import Any, Protocol
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class Brain(Protocol):
    model: str

    def respond(self, user_text: str, **context: Any) -> Any: ...


class ChatTransport(Protocol):
    def open(self, request: Request, *, timeout: float) -> Any: ...


class HTTPChatTransport:
    def open(self, request: Request, *, timeout: float) -> Any:
        return urlopen(request, timeout=timeout)


def chat_configuration(api_key: str | None = None, endpoint: str | None = None) -> tuple[str, str | None]:
    base = os.environ.get("MARCUS_CHAT_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
    url = endpoint or base + "/chat/completions"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Invalid chat API endpoint")
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and not local:
        raise ValueError("Remote chat API endpoints require HTTPS")
    key = api_key or os.environ.get("MARCUS_CHAT_API_KEY")
    if not key and parsed.hostname == "openrouter.ai":
        key = os.environ.get("OPENROUTER_API_KEY")
    if not key and not local:
        raise ValueError("No chat API key is configured")
    return url, key
