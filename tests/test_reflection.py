import tempfile
import threading
import unittest
from pathlib import Path

from marcus.events import AuditLog, read_events
from marcus.reflection import BackgroundReflector, ReflectionResult
from marcus.style import StyleProfileStore, StyleProposal


class BlockingReflectionClient:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def reflect(self, state, *, turn_id):
        self.started.set()
        self.release.wait(timeout=2)
        return ReflectionResult(
            memory_decisions=[],
            style=StyleProposal(False, [], [], 0.2, "Not enough evidence"),
        )


class EmptyMemoryCurator:
    def apply(self, decisions, *, retrieved_ids):
        return []


class BackgroundReflectorTests(unittest.TestCase):
    def test_submit_does_not_wait_for_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = AuditLog(root / "logs")
            client = BlockingReflectionClient()
            reflector = BackgroundReflector(
                client,
                EmptyMemoryCurator(),
                StyleProfileStore(root / "vault"),
                audit=audit,
            )

            reflector.submit(
                {"recent_history": [{"role": "user", "content": "hello"}]},
                active_memory_ids=set(),
                turn_id="turn-test",
            )

            self.assertTrue(client.started.wait(timeout=1))
            events = read_events(audit.path, limit=20)
            self.assertIn("curator.queued", [event["event"] for event in events])
            self.assertIn("curator.started", [event["event"] for event in events])

            client.release.set()
            reflector.close()
            events = read_events(audit.path, limit=20)
            self.assertIn("curator.completed", [event["event"] for event in events])


if __name__ == "__main__":
    unittest.main()
