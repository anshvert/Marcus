from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from marcus.agents.manager import MemoryDecision
from marcus.memory.curation import MemoryCurator
from marcus.memory.store import MarkdownMemoryStore


def decision(**overrides):
    values = {
        "action": "create",
        "content": "The user prefers concise technical explanations",
        "target_id": None,
        "category": "preference",
        "confidence": 0.95,
        "sensitivity": "low",
        "rationale": "Stable preference",
    }
    values.update(overrides)
    return MemoryDecision(**values)


class MemoryCuratorTests(unittest.TestCase):
    def test_high_confidence_durable_memory_is_stored(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            result = MemoryCurator(store).apply([decision()], retrieved_ids=set())[0]
            self.assertEqual("created", result.status)
            self.assertEqual(1, len(store.list()))

    def test_sensitive_and_low_confidence_candidates_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            results = MemoryCurator(store).apply(
                [decision(sensitivity="high"), decision(confidence=0.3)],
                retrieved_ids=set(),
            )
            self.assertEqual(["rejected", "rejected"], [item.status for item in results])
            self.assertEqual([], store.list())

    def test_supersede_requires_retrieved_target(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            old = store.remember("The user prefers short answers", kind="preference")
            update = decision(
                action="supersede",
                target_id=old.id,
                content="The user now prefers detailed answers",
            )

            rejected = MemoryCurator(store).apply([update], retrieved_ids=set())[0]
            self.assertEqual("rejected", rejected.status)

            accepted = MemoryCurator(store).apply([update], retrieved_ids={old.id})[0]
            self.assertEqual("superseded", accepted.status)
            self.assertEqual(old.id, accepted.memory.supersedes)


if __name__ == "__main__":
    unittest.main()
