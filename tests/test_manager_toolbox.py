import tempfile
import unittest
from pathlib import Path

from marcus.memory.store import MarkdownMemoryStore
from marcus.observability.audit import AuditLog
from marcus.tools.jev import JevDecision
from marcus.tools.manager import MarcusToolbox
from marcus.core.policy import CapabilityPolicy


class FakeDecisionTool:
    model = "typesafe/jev-1.13"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, text, *, history, turn_id=None):
        self.calls += 1
        return JevDecision(
            intent="general",
            confidence=0.9,
            needs_memory=False,
            needs_knowledge=False,
            needs_reflection=False,
            memory_probability=0.0,
            knowledge_probability=0.0,
            reflection_probability=0.0,
            source="jev",
        )


class FakeDataAgent:
    def handle(self, text, *, turn_id=None):
        raise AssertionError("data agent should not be called")


class FakeKnowledgeStore:
    embeddings = None

    def __init__(self, documents=None):
        self.documents = documents or []

    def search(self, query, *, limit, turn_id=None):
        return []

    def list_documents(self):
        return self.documents


class MarcusToolboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.decision = FakeDecisionTool()
        self.toolbox = MarcusToolbox(
            decision_tool=self.decision,
            data_agent=FakeDataAgent(),
            memory=MarkdownMemoryStore(root / "vault"),
            knowledge=FakeKnowledgeStore(),
            audit=AuditLog(root / "logs"),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_jev_is_not_called_when_a_turn_begins(self) -> None:
        self.toolbox.begin_turn("tell me a joke", [])

        self.assertEqual(self.decision.calls, 0)

    def test_disabled_actions_are_hidden_and_denied_at_execution(self):
        self.toolbox.policy = CapabilityPolicy({"runtime_status"})
        actions = self.toolbox.schemas[0]["function"]["parameters"]["properties"]["action"]["enum"]
        self.assertEqual(actions, ["runtime_status"])
        result = self.toolbox.execute("marcus_action", {"action": "save_memory", "input": "I prefer tea"})
        self.assertEqual(result["status"], "denied")
        self.assertEqual(self.toolbox.memory.list(), [])

    def test_model_cannot_replace_user_supplied_path_on_delegation(self):
        received = []

        class Data:
            def handle(self, text, **kwargs):
                from marcus.agents.data import DataAgentResponse
                received.append(text)
                return DataAgentResponse("ok", {})

        self.toolbox.data_agent = Data()
        original = 'Ingest my resume at "/tmp/resume.md"'
        self.toolbox.begin_turn(original, [])
        self.toolbox.execute("marcus_action", {"action": "delegate_data", "input": 'Ingest "/tmp/private.md"'})
        self.assertEqual(received, [original])

    def test_jev_runs_only_when_manager_requests_it(self) -> None:
        self.toolbox.begin_turn("something ambiguous", [])

        result = self.toolbox.execute(
            "marcus_action",
            {"action": "ask_jev", "input": "something ambiguous"},
            turn_id="turn-1",
        )

        self.assertEqual(self.decision.calls, 1)
        self.assertEqual(result["intent"], "general")

    def test_explicit_natural_language_memory_action_is_saved(self) -> None:
        self.toolbox.begin_turn("remember that I prefer tea", [])

        result = self.toolbox.execute(
            "marcus_action",
            {"action": "save_memory", "input": "I prefer tea"},
            turn_id="turn-2",
        )

        self.assertEqual(result["status"], "saved")
        self.assertTrue(self.toolbox.suppress_reflection)
        self.assertEqual(self.toolbox.memory.list()[0].content, "I prefer tea")

    def test_source_catalog_exposes_titles_without_document_contents(self) -> None:
        self.toolbox.knowledge = FakeKnowledgeStore(
            [
                {
                    "title": "Ansh's resume",
                    "document_type": "resume",
                    "source_name": "resume.pdf",
                }
            ]
        )

        catalog = self.toolbox.source_catalog()

        self.assertIn("Ansh's resume", catalog)
        self.assertIn("resume.pdf", catalog)
        self.assertNotIn("Education", catalog)


if __name__ == "__main__":
    unittest.main()
