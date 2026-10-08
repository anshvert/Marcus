from __future__ import annotations

import os


def bounded_history(history: list[dict[str, str]], *, max_chars: int | None = None) -> list[dict[str, str]]:
    budget = max_chars if max_chars is not None else max(1000, int(os.environ.get("MARCUS_HISTORY_CHARS", "12000")))
    messages: list[dict[str, str]] = []
    used = 0
    for message in reversed(history[-12:]):
        size = len(str(message.get("content", "")))
        if used + size > budget:
            break
        messages.insert(0, message)
        used += size
    while messages and messages[0].get("role") != "user":
        messages.pop(0)
    return messages
