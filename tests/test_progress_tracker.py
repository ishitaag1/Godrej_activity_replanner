from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

import progress_tracker
import scheduler as sch
from tests.test_scheduler import four_activity_frame


class TestProgressTracker(unittest.TestCase):
    def test_unknown_actual_progress_is_not_inferred_from_calendar(self) -> None:
        frame = four_activity_frame()
        report = progress_tracker.build_progress_report(
            frame, sch.compute_cpm(frame), date(2026, 3, 9)
        )
        self.assertTrue(report["actual_progress_pct"].isna().all())
        self.assertEqual(set(report["status"]), {"Not started", "Overdue"})

    def test_actual_progress_and_overdue_calculations(self) -> None:
        frame = four_activity_frame()
        frame["actual_progress_pct"] = None
        frame["actual_start"] = ""
        frame["actual_finish"] = ""
        frame.loc[frame["activity_id"] == "B", "actual_start"] = "2026-03-04"
        frame.loc[frame["activity_id"] == "B", "actual_progress_pct"] = 50
        report = progress_tracker.build_progress_report(
            frame, sch.compute_cpm(frame), date(2026, 3, 8)
        ).set_index("activity_id")
        self.assertEqual(report.at["B", "status"], "Overdue")
        self.assertEqual(report.at["B", "remaining_duration_days"], 1)
        self.assertEqual(report.at["B", "actual_progress_pct"], 50)

    def test_one_hundred_percent_actual_progress_does_not_fabricate_finish_date(self) -> None:
        frame = four_activity_frame()
        frame["actual_progress_pct"] = None
        frame["actual_start"] = ""
        frame["actual_finish"] = ""
        frame.loc[frame["activity_id"] == "A", "actual_progress_pct"] = 100
        report = progress_tracker.build_progress_report(
            frame, sch.compute_cpm(frame), date(2026, 3, 2)
        ).set_index("activity_id")
        self.assertEqual(report.at["A", "status"], "Completed")
        self.assertTrue(pd.isna(report.at["A", "forecast_finish"]))


if __name__ == "__main__":
    unittest.main()
