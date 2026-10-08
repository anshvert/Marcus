from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from marcus.memory.store import MarkdownMemoryStore


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

    def test_recall_ignores_generic_question_words(self) -> None:
        with TemporaryDirectory() as directory:
            store = MarkdownMemoryStore(Path(directory))
            store.remember("Prefers answers without mentioning where information came from")

            self.assertEqual([], store.recall("Where did I graduate from?"))

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

    def test_new_facts_share_daily_file_and_external_edits_reindex(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = MarkdownMemoryStore(root)
            one = store.remember("The user likes tea", kind="profile")
            two = store.remember("The user likes Python", kind="preference")
            self.assertEqual(len(list(store.daily_dir.glob("*.md"))), 1)
            self.assertEqual(len(list(store.memory_dir.glob("*.md"))), 0)
            path = next(store.daily_dir.glob("*.md"))
            path.write_text(path.read_text().replace("likes tea", "likes coffee"))
            self.assertEqual(store.recall("coffee")[0].id, one.id)
            self.assertEqual(store.recall("tea"), [])
            self.assertTrue(store.forget(one.id))
            self.assertEqual([item.id for item in store.list()], [two.id])
            reopened = MarkdownMemoryStore(root)
            self.assertEqual(reopened.list()[0].id, two.id)
            self.assertNotIn("coffee", (root / "Marcus" / "USER.md").read_text())

    def test_legacy_notes_are_preserved_and_index_can_be_rebuilt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = MarkdownMemoryStore(root)
            saved = store.remember("Prefers clear documentation")
            legacy = store.memory_dir / f"{saved.id}.md"
            original = store._serialize(saved)
            legacy.write_text(original)
            next(store.daily_dir.glob("*.md")).unlink()
            store.sync()
            self.assertEqual(store.recall("documentation")[0].id, saved.id)
            self.assertEqual(legacy.read_text(), original)
            store.index_path.unlink()
            rebuilt = MarkdownMemoryStore(root)
            self.assertEqual(rebuilt.recall("documentation")[0].id, saved.id)

    def test_bad_manual_note_does_not_break_retrieval_or_overwrite_user_profile(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = MarkdownMemoryStore(root)
            (root / "Marcus" / "USER.md").write_text("My own profile")
            (store.memory_dir / "broken.md").write_text("A hand-written note without metadata")
            saved = store.remember("The user prefers readable code", kind="preference")
            self.assertEqual(store.recall("readable")[0].id, saved.id)
            self.assertTrue(store.sync_errors)
            self.assertEqual((root / "Marcus" / "USER.md").read_text(), "My own profile")


if __name__ == "__main__":
    unittest.main()
