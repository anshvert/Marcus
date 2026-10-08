from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
import tempfile
import struct
from concurrent.futures import Future, ThreadPoolExecutor, CancelledError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Callable

from marcus.observability.audit import AuditLog, current_session


def _plain_speech(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " Code is shown in the chat. ", text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_#>]", "", text)
    cleaned = " ".join(text.split())
    return cleaned if any(character.isalnum() for character in cleaned) else ""


def _audio_has_frames(path: str) -> bool:
    """`say` can return zero with empty audio when its service is unavailable."""
    with open(path, "rb") as stream:
        header = stream.read(12)
        if header[:4] != b"FORM" or header[8:12] not in {b"AIFF", b"AIFC"}:
            return False
        for _ in range(64):
            chunk = stream.read(8)
            if len(chunk) != 8:
                return False
            name, size = struct.unpack(">4sI", chunk)
            if name == b"COMM":
                content = stream.read(min(size, 6))
                return len(content) == 6 and struct.unpack(">I", content[2:6])[0] > 0
            stream.seek(size + size % 2, 1)
    return False


class SpeechChunker:
    """Buffer complete sentences, keeping times, decimals, and URLs intact."""

    def __init__(self, *, minimum: int = 60, maximum: int = 320) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        chunks: list[str] = []
        while True:
            boundary = self._boundary()
            if boundary is None:
                break
            chunk = _plain_speech(self.buffer[:boundary])
            self.buffer = self.buffer[boundary:]
            if chunk:
                chunks.append(chunk)
        return chunks

    def finish(self) -> list[str]:
        chunk = _plain_speech(self.buffer)
        self.buffer = ""
        return [chunk] if chunk else []

    def reset(self) -> None:
        self.buffer = ""

    def _boundary(self) -> int | None:
        fenced = False
        for index, character in enumerate(self.buffer):
            if self.buffer.startswith("```", index):
                fenced = not fenced
            if fenced:
                continue
            length = index + 1
            if character not in ".!?":
                continue
            # Require lookahead: punctuation at a token boundary may be part
            # of a decimal, URL, abbreviation, or a trailing closing quote.
            end = length
            while end < len(self.buffer) and self.buffer[end] in '\"\u201d\u2019\')]}!?':
                end += 1
            if end >= len(self.buffer) or not self.buffer[end].isspace():
                continue
            token = self.buffer[:length].rsplit(None, 1)[-1].casefold()
            if character == "." and (token in {"dr.", "mr.", "mrs.", "ms.", "prof.", "e.g.", "i.e.", "a.m.", "p.m."}
                                      or re.fullmatch(r"(?:[a-z]\.)+", token)):
                continue
            if length >= 3 and not re.fullmatch(r"\s*\d+\.", self.buffer[:length]):
                return end
        if len(self.buffer) < self.maximum:
            return None
        # Avoid cutting code fences or splitting words / numbers just because
        # the model happened to deliver a large token fragment.
        if self.buffer.count("```") % 2:
            return None
        space = self.buffer.rfind(" ", 0, self.maximum)
        return space + 1 if space >= self.minimum else None


@dataclass(frozen=True)
class SpeechItem:
    turn_id: str | None
    text: str
    generation: int
    queued_at: float
    audio: Future | None = None
    session_id: str | None = None


class SpeechOutput:
    """Prefetch system TTS audio while previous sentences play in order."""

    def __init__(
        self,
        *,
        audit: AuditLog | None = None,
        command: str | None = None,
        runner: Callable[[list[str]], subprocess.Popen] | None = None,
        player: str | None = None,
    ) -> None:
        self.audit = audit
        configured = os.environ.get("MARCUS_VOICE", "1").casefold()
        self.enabled = configured not in {"0", "false", "no", "off"}
        self.command = command or shutil.which("say")
        self.player = player or (shutil.which("afplay") if runner is None else None)
        self.mode = os.environ.get("MARCUS_VOICE_MODE", "stream").casefold()
        if self.mode not in {"stream", "buffered"}:
            raise ValueError("MARCUS_VOICE_MODE must be stream or buffered")
        self.voice = os.environ.get("MARCUS_VOICE_NAME", "").strip()
        self.rate = os.environ.get("MARCUS_VOICE_RATE", "195").strip()
        self.chunker = SpeechChunker()
        self._queue: queue.Queue[SpeechItem | None] = queue.Queue()
        self._lock = threading.RLock()
        self._generation = 0
        self._closed = False
        self._synth_processes: set[subprocess.Popen] = set()
        self._audio_files = tempfile.TemporaryDirectory(prefix="marcus-speech-")
        self._synthesis = ThreadPoolExecutor(max_workers=2, thread_name_prefix="marcus-tts")
        self._current: subprocess.Popen | None = None
        self._runner = runner or self._start_process
        self._thread: threading.Thread | None = None

        if self.enabled and self.command:
            self._thread = threading.Thread(
                target=self._worker,
                name="marcus-speech",
                daemon=True,
            )
            self._thread.start()
            self._emit("voice.ready", engine=self.command, rate=self.rate, status="ready")
        else:
            reason = "disabled" if not self.enabled else "system voice engine not found"
            self.enabled = False
            self._emit("voice.unavailable", summary=reason, status="unavailable")

    def feed(self, text: str, *, turn_id: str | None = None) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            if self.mode == "buffered":
                self.chunker.buffer += text
            else:
                for chunk in self.chunker.feed(text):
                    self._enqueue(chunk, turn_id=turn_id)

    def finish(self, *, turn_id: str | None = None) -> None:
        if not self.enabled or self._closed:
            return
        with self._lock:
            for chunk in self.chunker.finish():
                self._enqueue(chunk, turn_id=turn_id)

    def interrupt(self, *, turn_id: str | None = None) -> None:
        interrupted = False
        with self._lock:
            self._generation += 1
            self.chunker.reset()
            for process in list(self._synth_processes):
                if process.poll() is None:
                    process.terminate()
                    interrupted = True
            if self._current is not None and self._current.poll() is None:
                self._current.terminate()
                interrupted = True
        while True:
            try:
                item = self._queue.get_nowait()
                if item is not None and item.audio:
                    item.audio.cancel()
                self._queue.task_done()
                interrupted = True
            except queue.Empty:
                break
        if interrupted:
            self._emit("voice.interrupted", turn_id=turn_id, status="interrupted")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.interrupt()
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=5)
        self._synthesis.shutdown(wait=True)
        self._audio_files.cleanup()
        self._thread = None

    def drain(self) -> None:
        """Finish queued speech before a graceful conversational goodbye."""
        self._queue.join()

    def _enqueue(self, text: str, *, turn_id: str | None) -> None:
        self._emit(
            "voice.queued",
            turn_id=turn_id,
            text=text,
            character_count=len(text),
            status="queued",
        )
        generation = self._generation
        audio = self._synthesis.submit(copy_context().run, self._synthesize, text, turn_id, generation) if self.player else None
        self._queue.put(SpeechItem(turn_id, text, generation, time.monotonic(), audio, current_session()))

    def _synthesize(self, text: str, turn_id: str | None, generation: int) -> str:
        started = time.monotonic()
        handle = tempfile.NamedTemporaryFile(suffix=".aiff", dir=self._audio_files.name, delete=False)
        path = handle.name
        handle.close()
        self._emit("voice.synthesis.started", turn_id=turn_id, character_count=len(text), status="running")
        process = None
        try:
            with self._lock:
                if generation != self._generation:
                    raise CancelledError()
                arguments = self._arguments(text)
                arguments[1:1] = ["-o", path]
                process = self._runner(arguments)
                self._synth_processes.add(process)
            if process.wait() != 0 or not _audio_has_frames(path):
                raise RuntimeError("System speech synthesis failed")
            self._emit("voice.synthesis.completed", turn_id=turn_id,
                       duration_ms=round((time.monotonic() - started) * 1000), status="success")
            return path
        except Exception:
            self._emit("voice.synthesis.failed", turn_id=turn_id,
                       duration_ms=round((time.monotonic() - started) * 1000), status="failed")
            raise
        finally:
            with self._lock:
                if process:
                    self._synth_processes.discard(process)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            started = time.monotonic()
            status = "success"
            path = None
            try:
                if item.audio:
                    path = item.audio.result()
                with self._lock:
                    if item.generation != self._generation:
                        raise CancelledError()
                    self._emit("voice.started", turn_id=item.turn_id, text=item.text,
                               session_id=item.session_id,
                               queue_wait_ms=round((time.monotonic() - item.queued_at) * 1000),
                               status="speaking")
                    started = time.monotonic()
                    arguments = [self.player, path] if path else self._arguments(item.text)
                    process = self._runner(arguments)
                    self._current = process
                return_code = process.wait()
                status = "interrupted" if item.generation != self._generation else ("success" if return_code == 0 else "failed")
            except CancelledError:
                status = "interrupted"
            except Exception:
                status = "failed"
            finally:
                with self._lock:
                    self._current = None
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                self._emit(
                    "voice.completed",
                    turn_id=item.turn_id,
                    session_id=item.session_id,
                    duration_ms=round((time.monotonic() - started) * 1000),
                    status=status,
                )
                self._queue.task_done()

    def _arguments(self, text: str) -> list[str]:
        arguments = [self.command or "say"]
        if self.voice:
            arguments.extend(["-v", self.voice])
        if self.rate:
            arguments.extend(["-r", self.rate])
        arguments.extend(["--", text])
        return arguments

    @staticmethod
    def _start_process(arguments: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            arguments,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _emit(self, event: str, *, turn_id: str | None = None, **data: object) -> None:
        if self.audit:
            self.audit.emit(event, turn_id=turn_id, **data)
