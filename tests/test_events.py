import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from marcus.observability.audit import AuditLog, format_event, read_events


class AuditLogTests(unittest.TestCase):
    def test_writes_jsonl_and_redacts_secrets(self) -> None:
        with TemporaryDirectory() as directory:
            audit = AuditLog(Path(directory), session_id="session-test")
            audit.emit(
                "model.request.started",
                turn_id="turn-test",
                api_key="do-not-log-this",
                nested={"authorization": "secret", "safe": "visible"},
            )

            event = read_events(audit.path, limit=1)[0]
            self.assertEqual("model.request.started", event["event"])
            self.assertEqual("[REDACTED]", event["data"]["api_key"])
            self.assertEqual("[REDACTED]", event["data"]["nested"]["authorization"])
            self.assertEqual("visible", event["data"]["nested"]["safe"])
            json.dumps(event)

    def test_normal_trace_shows_exact_prompt_and_timing(self) -> None:
        record = {
            "timestamp": "2026-10-07T12:34:56+00:00",
            "event": "model.request.started",
            "data": {
                "duration_ms": 125,
                "messages": [{"role": "system", "content": "line one\nline two"}],
                "tools": [{"type": "function", "function": {"name": "ingest"}}],
            },
        }

        rendered = format_event(record, include_payloads=True)
        concise = format_event(record)

        self.assertIn("+125ms", rendered)
        self.assertIn("line one\n    line two", rendered)
        self.assertIn('"name": "ingest"', rendered)
        self.assertNotIn("line one", concise)


if __name__ == "__main__":
    unittest.main()
