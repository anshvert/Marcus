import unittest

from marcus.policy import ActionPolicy, Risk


class ActionPolicyTests(unittest.TestCase):
    def test_read_only_action_is_allowed(self) -> None:
        decision = ActionPolicy().evaluate(Risk.READ_ONLY)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.confirmation_required)

    def test_external_action_requires_confirmation(self) -> None:
        decision = ActionPolicy().evaluate(Risk.EXTERNAL)
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.confirmation_required)

    def test_confirmation_allows_external_action(self) -> None:
        decision = ActionPolicy().evaluate(Risk.EXTERNAL, user_confirmed=True)
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
