import unittest

from marcus.tools.jev import JevDecisionTool


class JevDecisionToolTests(unittest.TestCase):
    def test_parses_casual_intent_without_enabling_context_subsystems(self) -> None:
        tool = JevDecisionTool(api_key="unused")

        decision = tool._parse(
            {
                "answers": {
                    "intent": {
                        "choice": "casual",
                        "confidence": 0.96,
                        "probabilities": {"casual": 0.97, "general": 0.03},
                    },
                    "needs_memory": {"noul": 0.03},
                    "needs_knowledge": {"noul": 0.01},
                    "needs_reflection": {"noul": 0.08},
                }
            }
        )

        self.assertEqual(decision.intent, "casual")
        self.assertEqual(decision.source, "jev")
        self.assertFalse(decision.needs_memory)
        self.assertFalse(decision.needs_knowledge)
        self.assertFalse(decision.needs_reflection)

    def test_parses_data_agent_intent(self) -> None:
        tool = JevDecisionTool(api_key="unused")

        decision = tool._parse(
            {
                "answers": {
                    "intent": {
                        "choice": "data_agent",
                        "confidence": 0.98,
                        "probabilities": {"data_agent": 0.99, "general": 0.01},
                    },
                    "needs_memory": {"noul": 0.02},
                    "needs_knowledge": {"noul": 0.1},
                    "needs_reflection": {"noul": 0.01},
                }
            }
        )

        self.assertEqual(decision.intent, "data_agent")
        self.assertEqual(decision.source, "jev")

    def test_parses_typed_jev_answers(self) -> None:
        tool = JevDecisionTool(api_key="unused", threshold=0.65)

        decision = tool._parse(
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

    def test_unavailable_jev_has_one_neutral_fallback_not_phrase_rules(self) -> None:
        tool = JevDecisionTool(api_key="unused")

        greeting = tool._fallback()
        ingestion = tool._fallback()

        self.assertEqual(greeting.intent, "general")
        self.assertEqual(ingestion.intent, "general")
        self.assertFalse(hasattr(tool, "_deterministic"))


if __name__ == "__main__":
    unittest.main()
