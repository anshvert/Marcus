import unittest

from marcus.jev import JevRouter


class JevRouterTests(unittest.TestCase):
    def test_obvious_greeting_uses_zero_api_fast_path(self) -> None:
        router = JevRouter(api_key=None)

        decision = router.classify("yo whts up")

        self.assertEqual(decision.intent, "social")
        self.assertEqual(decision.source, "deterministic")
        self.assertFalse(decision.needs_memory)
        self.assertFalse(decision.needs_knowledge)
        self.assertFalse(decision.needs_reflection)

    def test_explicit_ingestion_routes_to_data_agent_without_api(self) -> None:
        router = JevRouter(api_key=None)

        decision = router.classify('ingest my resume at "/tmp/resume.pdf"')

        self.assertEqual(decision.intent, "data_agent")
        self.assertEqual(decision.source, "deterministic")

    def test_parses_typed_jev_answers(self) -> None:
        router = JevRouter(api_key="unused", threshold=0.65)

        decision = router._parse(
            {
                "answers": {
                    "intent": {
                        "choice": "knowledge",
                        "confidence": 0.93,
                        "probabilities": {"knowledge": 0.9, "general": 0.1},
                    },
                    "needs_memory": {"noul": 0.2},
                    "needs_knowledge": {"noul": 0.91},
                    "needs_reflection": {"noul": 0.1},
                }
            }
        )

        self.assertEqual(decision.intent, "knowledge")
        self.assertTrue(decision.needs_knowledge)
        self.assertFalse(decision.needs_memory)
        self.assertFalse(decision.needs_reflection)


if __name__ == "__main__":
    unittest.main()
