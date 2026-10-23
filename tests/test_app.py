from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class TestDashboardStartup(unittest.TestCase):
    def test_dashboard_renders_navigation_and_overview(self) -> None:
        app_file = Path(__file__).resolve().parents[1] / "app.py"
        result = AppTest.from_file(str(app_file)).run(timeout=90)
        self.assertEqual(len(result.exception), 0)
        self.assertEqual(len(result.title), 1)
        self.assertEqual(len(result.tabs), 10)


if __name__ == "__main__":
    unittest.main()
