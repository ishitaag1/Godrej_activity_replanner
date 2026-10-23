"""Progress and forecast calculations based only on explicitly supplied progress."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

import scheduler as sch


def build_progress_report(
    frame: pd.DataFrame,
    schedule: sch.ProjectSchedule,
    as_of: date,
) -> pd.DataFrame:
    """Report actual status without inferring actual progress from planned dates."""
    source = frame.set_index("activity_id", drop=False)
    rows: list[dict[str, Any]] = []
    for activity_id, item in schedule.activities.items():
        row = source.loc[activity_id]
        actual_progress = row.get("actual_progress_pct")
        if pd.isna(actual_progress):
            actual_progress = None
        else:
            actual_progress = float(actual_progress)
            if not 0 <= actual_progress <= 100:
                raise sch.ScheduleError(
                    f"{activity_id} actual_progress_pct must be between 0 and 100."
                )
        actual_start = row.get("actual_start") or None
        actual_finish = row.get("actual_finish") or None
        if pd.isna(actual_start):
            actual_start = None
        if pd.isna(actual_finish):
            actual_finish = None
        if actual_finish:
            status = "Completed"
            remaining = 0
            forecast_finish = str(actual_finish)
        elif actual_progress is not None and actual_progress >= 100:
            status = "Completed"
            remaining = 0
            forecast_finish = None
        elif actual_progress is not None and actual_progress > 0:
            status = "In progress"
            remaining = math.ceil(item.duration_days * (100 - actual_progress) / 100)
            forecast_finish = (
                date.fromisoformat(str(actual_start)).isoformat()
                if remaining == 0 and actual_start
                else (as_of.fromordinal(as_of.toordinal() + remaining - 1)).isoformat()
            )
        elif actual_start:
            status = "In progress"
            remaining = item.duration_days
            forecast_finish = (as_of.fromordinal(as_of.toordinal() + remaining - 1)).isoformat()
        else:
            status = "Not started"
            remaining = item.duration_days
            forecast_finish = item.earliest_finish.isoformat()
        overdue = (
            actual_finish is None
            and as_of > item.earliest_finish
            and (actual_progress is None or actual_progress < 100)
        )
        planned = row.get("planned_progress_pct")
        planned = None if pd.isna(planned) else float(planned)
        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": item.activity_name,
                "tower_id": item.tower_id,
                "planned_progress_pct": planned,
                "actual_progress_pct": actual_progress,
                "actual_start": actual_start,
                "actual_finish": actual_finish,
                "remaining_duration_days": remaining,
                "forecast_finish": forecast_finish,
                "status": "Overdue" if overdue else status,
                "overdue": overdue,
                "at_risk": overdue or item.total_float_days <= 2,
                "is_milestone": bool(row.get("is_milestone", False)),
                "schedule_variance_days": (
                    (date.fromisoformat(str(actual_finish)) - item.earliest_finish).days
                    if actual_finish
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)
