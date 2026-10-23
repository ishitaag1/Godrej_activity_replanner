"""Deterministic delay scenarios and baseline-versus-forecast comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

import scheduler as sch


@dataclass(frozen=True)
class SavedScenario:
    """An immutable scenario result suitable for session-state storage."""

    name: str
    delays: dict[str, int]
    schedule: sch.ProjectSchedule
    impact: dict[str, Any]


def calculate_scenario(
    frame: pd.DataFrame, name: str, delay_events: dict[str, int]
) -> SavedScenario:
    """Apply all delay events together and preserve the input baseline."""
    scenario_name = name.strip()
    if not scenario_name:
        raise sch.ScheduleError("Scenario name cannot be empty.")
    _, _, revised, impact = sch.apply_delays(frame, delay_events)
    return SavedScenario(scenario_name, dict(delay_events), revised, impact)


def compare_schedules(
    schedules: dict[str, sch.ProjectSchedule],
) -> pd.DataFrame:
    """Create a compact comparison table from already-computed schedules."""
    if not schedules:
        raise sch.ScheduleError("At least one schedule is required for comparison.")
    rows = []
    for name, schedule in schedules.items():
        rows.append(
            {
                "scenario": name,
                "project_start": schedule.project_start.isoformat(),
                "project_finish": schedule.project_finish.isoformat(),
                "duration_days": schedule.project_duration_days,
                "critical_activity_count": len(schedule.critical_activities),
                "critical_path_count": len(schedule.critical_paths),
                "critical_path": " -> ".join(schedule.critical_path),
            }
        )
    return pd.DataFrame(rows)


def export_scenarios(
    baseline: sch.ProjectSchedule,
    scenarios: list[SavedScenario],
    export_format: str = "json",
) -> str:
    """Serialize saved scenario comparisons as JSON or CSV text."""
    comparison = compare_schedules(
        {"Baseline": baseline, **{item.name: item.schedule for item in scenarios}}
    )
    event_map = {item.name: item.delays for item in scenarios}
    comparison["delay_events"] = comparison["scenario"].map(
        lambda name: json.dumps(event_map.get(name, {}), sort_keys=True)
    )
    if export_format.lower() == "csv":
        return comparison.to_csv(index=False)
    if export_format.lower() != "json":
        raise sch.ScheduleError("Scenario export format must be 'json' or 'csv'.")
    return comparison.to_json(orient="records", indent=2)


def rule_based_recovery_review(schedule: sch.ProjectSchedule) -> list[dict[str, str]]:
    """Suggest review priorities only; this is not resource optimization."""
    return [
        {
            "activity_id": activity_id,
            "recommendation": (
                "Review this zero-float activity with the responsible team for "
                "feasible duration assumptions and constraint removal. Do not "
                "change mandatory construction sequence without approved review."
            ),
            "basis": "CPM criticality; no crew/resource capacity model is available.",
        }
        for activity_id in schedule.critical_activities
    ]


def schedule_completion_delta(
    before: sch.ProjectSchedule, after: sch.ProjectSchedule
) -> int:
    """Return the project completion slip in calendar days."""
    return (after.project_finish - before.project_finish).days
