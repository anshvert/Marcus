import unittest

from marcus.observability.web import TraceDashboardHandler


class TraceDashboardTests(unittest.TestCase):
    def test_dashboard_asset_is_packaged_next_to_server(self) -> None:
        html = TraceDashboardHandler.dashboard_path.read_text(encoding="utf-8")

        self.assertIn("execution trace", html)
        self.assertIn("EventSource('/events?limit=250')", html)


if __name__ == "__main__":
    unittest.main()
