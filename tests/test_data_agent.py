import unittest

from marcus.agents.data import DataAgent


class DataAgentTests(unittest.TestCase):
    def test_extracts_quoted_path_with_spaces(self) -> None:
        path = DataAgent._extract_path('please ingest "/tmp/My Resume.pdf"')

        self.assertEqual(path, "/tmp/My Resume.pdf")

    def test_infers_resume_document_type(self) -> None:
        kind = DataAgent._document_type("ingest this", "/tmp/Ansh Resume.pdf")

        self.assertEqual(kind, "resume")


if __name__ == "__main__":
    unittest.main()
