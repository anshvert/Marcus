import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from marcus.agents.manager import BrainResponse, TurnCancelled
from marcus.memory.store import MarkdownMemoryStore
from marcus.memory.style import StyleProfileStore
from marcus.observability.audit import AuditLog, read_events
from marcus.runtime.service import MarcusRuntime
from marcus.runtime.state import RuntimeStore


class FakeBrain:
    model = "fake"

    def __init__(self):
        self.histories = []
        self.started = threading.Event()
        self.release = None

    def respond(self, text, **context):
        self.histories.append(context["history"])
        self.started.set()
        if self.release:
            self.release.wait(2)
        if context["cancel"].is_set():
            raise TurnCancelled("cancelled")
        context["on_text"]("Reply: " + text)
        return BrainResponse("Reply: " + text, False, {})


class FakeKnowledge:
    embeddings = None

    def list_documents(self):
        return []


class FakeDecision:
    model = "fake-jeV"


def fake_runtime(root, brain):
    memory = MarkdownMemoryStore(root / "vault")
    return MarcusRuntime(audit=AuditLog(root / "logs"), memory=memory,
                         knowledge=FakeKnowledge(), data_agent=None, brain=brain,
                         style=StyleProfileStore(root / "vault"), reflector=None,
                         decision_tool=FakeDecision(), state=RuntimeStore(root / "runtime.sqlite"))


def await_job(runtime, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = runtime.state.job(job_id)
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        threading.Event().wait(0.01)
    raise AssertionError("Job did not finish")


class RuntimeTests(unittest.TestCase):
    def test_sessions_resume_and_do_not_leak_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = fake_runtime(root, FakeBrain())
            first.turn("tea", session_id="alice")
            first.close()
            brain = FakeBrain()
            second = fake_runtime(root, brain)
            try:
                second.turn("remember?", session_id="alice")
                second.turn("hello", session_id="bob")
                self.assertEqual(brain.histories[0][0]["content"], "tea")
                self.assertEqual(brain.histories[1], [])
                events = read_events(second.audit.path, limit=30)
                self.assertEqual(events[-1]["session_id"], "bob")
                self.assertNotIn("chat.delta", [event["event"] for event in events])
            finally:
                second.close()

    def test_running_and_queued_jobs_cancel_without_saving_a_reply(self):
        with TemporaryDirectory() as directory:
            brain = FakeBrain()
            brain.release = threading.Event()
            runtime = fake_runtime(Path(directory), brain)
            try:
                first = runtime.submit("one")
                self.assertTrue(brain.started.wait(1))
                second = runtime.submit("two")
                self.assertTrue(runtime.cancel(first["id"]))
                self.assertTrue(runtime.cancel(second["id"]))
                brain.release.set()
                self.assertEqual(await_job(runtime, first["id"])["status"], "cancelled")
                self.assertEqual(await_job(runtime, second["id"])["status"], "cancelled")
                self.assertEqual(runtime.state.history("default"), [])
            finally:
                brain.release.set()
                runtime.close()

    def test_successful_job_persists_result_and_crash_is_not_replayed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = fake_runtime(root, FakeBrain())
            try:
                job = runtime.submit("hi")
                finished = await_job(runtime, job["id"])
                self.assertEqual(finished["result"]["reply"], "Reply: hi")
                unfinished = runtime.state.create_job("default")
                runtime.state.update_job(unfinished, "running")
                self.assertEqual(runtime.state.recover(), 1)
                self.assertEqual(runtime.state.job(unfinished)["status"], "interrupted")
            finally:
                runtime.close()

    def test_runtime_lease_prevents_a_second_owner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = fake_runtime(root, FakeBrain())
            second = fake_runtime(root, FakeBrain())
            try:
                first.claim()
                with self.assertRaises(RuntimeError):
                    second.claim()
            finally:
                first.close()
                second.close()

    def test_session_id_rejects_paths(self):
        with TemporaryDirectory() as directory:
            store = RuntimeStore(Path(directory) / "runtime.sqlite")
            with self.assertRaises(ValueError):
                store.ensure_session("../../other")
