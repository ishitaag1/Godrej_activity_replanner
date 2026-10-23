from __future__ import annotations

import unittest

import pandas as pd

import scheduler as sch
from scenario_manager import calculate_scenario, compare_schedules, export_scenarios
from tests.test_scheduler import four_activity_frame


class TestMultipleDelayScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = four_activity_frame()
        self.original = self.frame.copy(deep=True)

    def test_simultaneous_delays_recalculate_network_once_and_preserve_input(self) -> None:
        _, baseline, revised, impact = sch.apply_delays(
            self.frame, {"B": 3, "C": 1}
        )
        self.assertEqual(baseline.project_finish.isoformat(), "2026-03-09")
        self.assertEqual(revised.project_finish.isoformat(), "2026-03-10")
        self.assertEqual(impact["project_finish_slip_days"], 1)
        self.assertEqual(impact["delay_events"], {"B": 3, "C": 1})
        self.assertEqual(revised.critical_path, ["A", "B", "D"])
        self.assertEqual(
            revised.critical_paths, [["A", "B", "D"], ["A", "C", "D"]]
        )
        pd.testing.assert_frame_equal(self.frame, self.original)

    def test_empty_unknown_negative_and_noninteger_scenarios_rejected(self) -> None:
        for events in ({}, {"missing": 1}, {"A": -1}, {"A": 1.5}):
            with self.subTest(events=events), self.assertRaises(sch.ScheduleError):
                sch.apply_delays(self.frame, events)

    def test_scenarios_compare_and_export(self) -> None:
        scenario = calculate_scenario(self.frame, "Scenario A", {"C": 2})
        baseline = sch.compute_cpm(self.frame)
        comparison = compare_schedules(
            {"Baseline": baseline, scenario.name: scenario.schedule}
        )
        self.assertEqual(len(comparison), 2)
        self.assertEqual(scenario.impact["project_finish_slip_days"], 2)
        self.assertIn("Scenario A", export_scenarios(baseline, [scenario], "json"))
        self.assertIn("project_finish", export_scenarios(baseline, [scenario], "csv"))


if __name__ == "__main__":
    unittest.main()
