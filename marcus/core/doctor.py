from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from marcus.core.config import load_dotenv, model_name


def diagnostics() -> list[dict]:
    """Read-only checks. No model calls, downloads, or credential output."""
    load_dotenv()
    base = os.environ.get("MARCUS_CHAT_API_BASE", "https://openrouter.ai/api/v1")
    local = base.startswith(("http://127.0.0.1:", "http://localhost:"))
    key_present = bool(os.environ.get("MARCUS_CHAT_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    voice = os.environ.get("MARCUS_VOICE", "1").casefold() not in {"0", "false", "off", "no"}
    return [
        {"check": "python", "ok": sys.version_info >= (3, 9), "detail": sys.version.split()[0]},
        {"check": "conversation", "ok": key_present or local, "detail": f"{model_name()} via {base}; credentials {'configured' if key_present else 'not configured'}"},
        {"check": "pdf", "ok": importlib.util.find_spec("pypdf") is not None, "detail": "pypdf"},
        {"check": "voice", "ok": not voice or bool(shutil.which("say")), "detail": "disabled" if not voice else "macOS say + afplay" if shutil.which("afplay") else "system voice unavailable or direct-speech fallback"},
        {"check": "identity", "ok": Path("MARCUS.md").is_file(), "detail": "MARCUS.md"},
        {"check": "skills", "ok": Path(os.environ.get("MARCUS_SKILL_DIR", "skills")).is_dir(), "detail": "local skills directory"},
    ]
