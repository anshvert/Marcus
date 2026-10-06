from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from marcus.memory import MarkdownMemoryStore


class MarkdownMemoryStoreTests(unittest.TestCase):
    def test_remember_recall_and_forget(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            saved = store.remember("I prefer concise technical explanations")

            results = store.recall("concise explanations")
            self.assertEqual([saved.id], [item.id for item in results])

            self.assertTrue(store.forget(saved.id))
            self.assertEqual([], store.recall("concise explanations"))
            self.assertTrue((store.archive_dir / f"{saved.id}.md").exists())

    def test_empty_memory_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            with self.assertRaises(ValueError):
                store.remember("   ")

    def test_supersede_preserves_lineage_and_archives_old_memory(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            old = store.remember("I prefer concise explanations", kind="preference")
            new = store.supersede(
                old.id,
                "I prefer detailed explanations",
                kind="preference",
            )

            self.assertEqual(old.id, new.supersedes)
            self.assertIsNone(store.get(old.id))
            self.assertTrue((store.archive_dir / f"{old.id}.md").exists())


if __name__ == "__main__":
    unittest.main()
