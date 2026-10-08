import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marcus.agents.coding import CodexAgent
from marcus.observability.audit import AuditLog, read_events


class FakeInput(io.StringIO):
    def close(self):
        self.submitted = self.getvalue()
        super().close()


class FakeCodex:
    def __init__(self):
        self.stdin = FakeInput()
        records = [
            {"type": "item.completed", "item": {"type": "reasoning", "text": "private reasoning"}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python -m unittest", "status": "completed"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Implemented and tested."}},
        ]
        self.stdout = io.StringIO("\n".join(json.dumps(record) for record in records))
        self.returncode = 0

    def poll(self):
        return 0


class CodingAgentTests(unittest.TestCase):
    def test_fixed_workspace_sandbox_argv_and_public_action_trace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audit = AuditLog(root / "logs")
            process = FakeCodex()
            agent = CodexAgent(root, audit=audit, command="codex")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-secret", "MARCUS_GATEWAY_TOKEN": "fake-gateway-secret"}), patch("marcus.agents.coding.subprocess.Popen", return_value=process) as launch:
                response = agent.handle("Build a notes app", turn_id="coding-test")
            self.assertEqual(response.result["status"], "completed")
            self.assertEqual(process.stdin.submitted, "Build a notes app")
            args, kwargs = launch.call_args
            self.assertIn("workspace-write", args[0])
            self.assertIn(str(root.resolve()), args[0])
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", args[0])
            self.assertNotIn("OPENROUTER_API_KEY", kwargs["env"])
            self.assertTrue(kwargs["start_new_session"])
            events = read_events(audit.path, limit=20)
            self.assertIn("coding_agent.progress", [item["event"] for item in events])
            self.assertNotIn("private reasoning", json.dumps(events))

    def test_missing_cli_reports_unavailable_without_launching(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("marcus.agents.coding.shutil.which", return_value=None):
                agent = CodexAgent(root, audit=AuditLog(root / "logs"))
            self.assertEqual(agent.handle("Code something").result["status"], "unavailable")
