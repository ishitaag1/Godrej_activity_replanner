from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import activity_generator as generator
import scheduler as sch


class TestActivityGenerator(unittest.TestCase):
    def test_demo_config_generates_large_multi_tower_schedule(self) -> None:
        config = generator.load_project_config()
        frame = generator.generate_schedule(config)
        self.assertGreater(len(frame), 500)
        self.assertEqual(frame["activity_id"].nunique(), len(frame))
        self.assertEqual(set(frame["tower_id"].unique()), {"SITE", "T1", "T2"})
        self.assertEqual(
            set(frame.loc[frame["floor_id"] != "", "floor_id"].unique()),
            {f"F{i:02d}" for i in range(1, 6)},
        )
        self.assertTrue(frame["assumption_flag"].str.contains("Illustrative").all())
        self.assertTrue(frame["actual_progress_pct"].isna().all())
        for tower in ("T1", "T2"):
            retained = frame.loc[
                frame["activity_id"] == f"{tower}-STAIR-S02-RETAINED-FINISH"
            ].iloc[0]
            common_finishes = frame.loc[
                (frame["tower_id"] == tower)
                & frame["activity_id"].str.contains("-LOBBY-LOBBY-WALL"),
                "activity_id",
            ]
            self.assertTrue(set(common_finishes).issubset(set(retained["predecessors"])))

    def test_generation_ids_and_dependencies_are_reproducible(self) -> None:
        config = {
            "project_start_date": "2026-01-05",
            "tower_count": 1,
            "floors_per_tower": 2,
            "units_per_floor": 1,
        }
        first = generator.generate_schedule(config)
        second = generator.generate_schedule(config)
        self.assertEqual(first["activity_id"].tolist(), second["activity_id"].tolist())
        self.assertEqual(first["predecessors"].tolist(), second["predecessors"].tolist())
        floor_two = first.set_index("activity_id").loc["T1-F02-SHELL-COLUMN-REBAR"]
        self.assertIn("T1-F01-SHELL-SLAB-POUR", floor_two["predecessors"])

    def test_optional_activities_only_generate_when_enabled(self) -> None:
        config = {
            "project_start_date": "2026-01-05",
            "tower_count": 1,
            "floors_per_tower": 1,
            "units_per_floor": 1,
        }
        without = generator.generate_schedule(config)
        config["optional_activities"] = ["BACKING-PLASTER", "WET-CEILING-FRAME"]
        with_options = generator.generate_schedule(config)
        self.assertFalse(without["activity_id"].str.contains("BACKING-PLASTER").any())
        self.assertTrue(with_options["activity_id"].str.contains("BACKING-PLASTER").any())
        self.assertTrue(
            with_options.loc[
                with_options["activity_id"].str.contains("BACKING-PLASTER"),
                "is_conditional",
            ].all()
        )

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(generator.ConfigurationError):
            generator.generate_schedule(
                {
                    "project_start_date": "2026-01-05",
                    "tower_count": 0,
                    "floors_per_tower": 2,
                }
            )

    def test_terrace_cannot_be_generated_without_required_external_clearances(self) -> None:
        with self.assertRaisesRegex(
            generator.ConfigurationError, "requires the external"
        ):
            generator.generate_schedule(
                {
                    "project_start_date": "2026-01-05",
                    "tower_count": 1,
                    "floors_per_tower": 1,
                    "applicable_work_packages": ["terrace"],
                }
            )

    def test_schedule_inclusive_finish_matches_cpm(self) -> None:
        frame = generator.generate_schedule(
            {
                "project_start_date": "2026-01-05",
                "tower_count": 1,
                "floors_per_tower": 1,
                "units_per_floor": 1,
            }
        )
        schedule = sch.compute_cpm(frame)
        self.assertEqual(schedule.project_start, date(2026, 1, 5))
        self.assertTrue(
            all(
                row["baseline_finish"] == schedule.activities[row["activity_id"]].earliest_finish
                for _, row in frame.iterrows()
            )
        )

    def test_normalized_register_csv_round_trip(self) -> None:
        frame = generator.generate_schedule(
            {
                "project_start_date": "2026-01-05",
                "tower_count": 1,
                "floors_per_tower": 1,
                "units_per_floor": 1,
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "activity-register.csv"
            generator.export_activity_register(frame, path)
            with path.open("rb") as uploaded:
                imported = generator.import_activity_register(uploaded)
        self.assertEqual(frame["activity_id"].tolist(), imported["activity_id"].tolist())
        self.assertEqual(frame["predecessors"].tolist(), imported["predecessors"].tolist())
        self.assertEqual(imported["activity_id"].nunique(), len(imported))

    def test_csv_import_rejects_missing_predecessor(self) -> None:
        frame = generator.generate_schedule(
            {
                "project_start_date": "2026-01-05",
                "tower_count": 1,
                "floors_per_tower": 1,
                "units_per_floor": 1,
            }
        )
        frame.at[0, "predecessors"] = ["MISSING-ID"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad-register.csv"
            generator.export_activity_register(frame, path)
            with path.open("rb") as uploaded:
                with self.assertRaises(generator.ConfigurationError):
                    generator.import_activity_register(uploaded)


if __name__ == "__main__":
    unittest.main()
