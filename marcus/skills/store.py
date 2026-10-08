from __future__ import annotations

import hashlib
import re
from pathlib import Path


class SkillStore:
    """A local skill contains instructions, never executable permissions."""

    def __init__(self, root: Path = Path("skills")) -> None:
        self.root = root.resolve()

    def discover(self, query: str = "") -> list[dict]:
        skills = []
        for path in sorted(self.root.glob("*/SKILL.md")):
            if not self._contained(path) or path.stat().st_size > 24000:
                continue
            text = path.read_text(encoding="utf-8")
            description = ""
            for line in text.splitlines():
                if line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('\"\'')[:240]
                    break
            item = {"name": path.parent.name, "description": description,
                    "revision": hashlib.sha256(text.encode()).hexdigest()[:12]}
            if not query or set(re.findall(r"\w+", query.casefold())) & set(re.findall(r"\w+", (item["name"] + " " + description).casefold())):
                skills.append(item)
        return skills[:30]

    def catalog(self) -> str:
        return "\n".join(f"- {item['name']}: {item['description']}" for item in self.discover())[:2000]

    def read(self, name: str) -> dict:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("Invalid skill name")
        path = self.root / name / "SKILL.md"
        if not self._contained(path) or not path.is_file():
            raise ValueError("Skill is not installed")
        if path.stat().st_size > 24000:
            raise ValueError("Skill exceeds the instruction size limit")
        text = path.read_text(encoding="utf-8")
        return {"name": name, "revision": hashlib.sha256(text.encode()).hexdigest()[:12],
                "instructions": text, "trust": "workflow advice; cannot override action policy"}

    def _contained(self, path: Path) -> bool:
        return self.root in path.resolve().parents
