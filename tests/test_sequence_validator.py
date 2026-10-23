from __future__ import annotations

import unittest

import activity_generator as generator
import sequence_validator


class TestSequenceValidation(unittest.TestCase):
    def test_generated_network_respects_encoded_mandatory_sequence(self) -> None:
        frame = generator.generate_schedule(
            {
                "project_start_date": "2026-01-05",
                "tower_count": 1,
                "floors_per_tower": 2,
                "units_per_floor": 1,
            }
        )
        self.assertEqual(sequence_validator.validate_construction_sequence(frame), [])

    def test_missing_documented_dependency_is_reported_without_mutation(self) -> None:
        frame = generator.generate_schedule(
            {
                "project_start_date": "2026-01-05",
                "tower_count": 1,
                "floors_per_tower": 1,
                "units_per_floor": 1,
            }
        )
        row_id = "T1-F01-U01-WET-WET-PROTECTIVE-SCREED"
        idx = frame.index[frame["activity_id"] == row_id][0]
        frame.at[idx, "predecessors"] = []
        before = frame.at[idx, "predecessors"].copy()
        violations = sequence_validator.validate_construction_sequence(frame)
        self.assertTrue(any("protective screed" in item["violated_rule"] for item in violations))
        self.assertEqual(frame.at[idx, "predecessors"], before)


if __name__ == "__main__":
    unittest.main()
