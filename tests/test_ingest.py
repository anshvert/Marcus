from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from marcus.ingest import DocumentIngestor
from marcus.knowledge import KnowledgeStore


class FakeEmbeddings:
    model = "fake/embedding"

    def embed(self, texts, *, input_type, turn_id=None):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0, 0.0] if "python" in lowered else [0.0, 1.0])
        return vectors


class DocumentIngestorTests(unittest.TestCase):
    def test_ingests_markdown_and_retrieves_a_chunk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "resume.md"
            source.write_text(
                "# Resume\n\n## Experience\nBuilt Python services and automation.\n",
                encoding="utf-8",
            )
            embeddings = FakeEmbeddings()
            store = KnowledgeStore(root / "knowledge.sqlite", embeddings=embeddings)
            ingestor = DocumentIngestor(
                store,
                vault_path=root / "vault",
                embeddings=embeddings,
            )

            result = ingestor.ingest(
                str(source),
                label="My Resume",
                document_type="resume",
            )

            self.assertEqual("indexed", result.embedding_status)
            self.assertGreaterEqual(result.chunk_count, 1)
            self.assertTrue(Path(result.extracted_path).exists())
            hits = store.search("What Python experience do I have?")
            self.assertTrue(hits)
            self.assertIn("Python", hits[0].text)

    def test_duplicate_source_is_not_reingested(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.md"
            source.write_text("# Notes\n\nA durable reference note.", encoding="utf-8")
            store = KnowledgeStore(root / "knowledge.sqlite")
            ingestor = DocumentIngestor(store, vault_path=root / "vault")

            first = ingestor.ingest(str(source))
            second = ingestor.ingest(str(source))

            self.assertFalse(first.duplicate)
            self.assertTrue(second.duplicate)
            self.assertEqual(first.document_id, second.document_id)


if __name__ == "__main__":
    unittest.main()
