from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marcus.observability.audit import AuditLog


class EmbeddingError(RuntimeError):
    pass


class OpenRouterEmbeddings:
    endpoint = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EmbeddingError("OPENROUTER_API_KEY is not configured")
        self.model = model or os.environ.get(
            "MARCUS_EMBEDDING_MODEL",
            "qwen/qwen3-embedding-8b",
        )
        self.audit = audit

    def embed(
        self,
        texts: list[str],
        *,
        input_type: str,
        turn_id: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        started = time.monotonic()
        if self.audit:
            self.audit.emit(
                "embedding.request.started",
                turn_id=turn_id,
                model=self.model,
                endpoint=self.endpoint,
                item_count=len(texts),
                input_type=input_type,
                inputs=texts,
                status="running",
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Marcus Personal Agent",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            self._log_error(turn_id, started, f"HTTP {exc.code}")
            raise EmbeddingError(f"OpenRouter embeddings returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            self._log_error(turn_id, started, type(exc).__name__)
            raise EmbeddingError(f"Could not reach OpenRouter embeddings: {exc}") from exc

        try:
            ordered = sorted(result["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as exc:
            self._log_error(turn_id, started, "unexpected_response")
            raise EmbeddingError("OpenRouter returned an unexpected embeddings response") from exc

        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding count did not match input count")
        if self.audit:
            self.audit.emit(
                "embedding.request.completed",
                turn_id=turn_id,
                model=self.model,
                endpoint=self.endpoint,
                item_count=len(vectors),
                dimensions=len(vectors[0]) if vectors else 0,
                duration_ms=round((time.monotonic() - started) * 1000),
                usage=result.get("usage", {}),
                status="success",
            )
        return vectors

    def _log_error(self, turn_id: str | None, started: float, summary: str) -> None:
        if self.audit:
            self.audit.emit(
                "embedding.request.failed",
                turn_id=turn_id,
                model=self.model,
                duration_ms=round((time.monotonic() - started) * 1000),
                summary=summary,
                status="failed",
            )
