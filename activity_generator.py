"""Generate reproducible, editable construction schedules from sequence templates."""

from __future__ import annotations

import json
import ast
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import scheduler as sch

ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = ROOT / "templates" / "construction_sequence.json"
DEFAULT_CONFIG = ROOT / "data" / "demo_project_config.json"

ACTIVITY_COLUMNS = [
    "activity_id",
    "activity_name",
    "parent_activity_id",
    "phase",
    "work_package",
    "tower_id",
    "floor_id",
    "unit_id",
    "trade",
    "predecessors",
    "successors",
    "duration_days",
    "baseline_start",
    "baseline_finish",
    "actual_start",
    "actual_finish",
    "planned_progress_pct",
    "actual_progress_pct",
    "responsible_team",
    "risk_level",
    "activity_type",
    "is_milestone",
    "is_conditional",
    "applicability_condition",
    "source_document_reference",
    "assumption_flag",
    "data_classification",
]


class ConfigurationError(ValueError):
    """Raised when a project configuration cannot generate a valid schedule."""


def load_project_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the reproducible synthetic demonstration configuration."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read project configuration: {exc}") from exc


def load_templates(path: str | Path = DEFAULT_TEMPLATE) -> dict[str, Any]:
    """Read the activity master and sequence packages."""
    try:
        templates = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read activity templates: {exc}") from exc
    if not isinstance(templates.get("packages"), dict):
        raise ConfigurationError("Template file must contain a packages object.")
    return templates


def _validate_config(
    config: dict[str, Any], packages: dict[str, Any]
) -> tuple[date, list[str], int, int, int]:
    try:
        start_date = datetime.strptime(
            str(config["project_start_date"]), "%Y-%m-%d"
        ).date()
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("project_start_date must use YYYY-MM-DD.") from exc

    tower_count = config.get("tower_count")
    floors = config.get("floors_per_tower")
    units = config.get("units_per_floor", 1)
    staircases = config.get("staircase_count_per_tower", 2)
    for field_name, value, maximum in (
        ("tower_count", tower_count, 20),
        ("floors_per_tower", floors, 200),
        ("units_per_floor", units, 100),
        ("staircase_count_per_tower", staircases, 20),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ConfigurationError(f"{field_name} must be a whole number from 1 to {maximum}.")

    if config.get("calendar", "calendar_days") != "calendar_days":
        raise ConfigurationError(
            "Only calendar_days is supported; weekends and holidays are counted."
        )
    applicable = config.get("applicable_work_packages", list(packages))
    if (
        not isinstance(applicable, list)
        or not applicable
        or any(not isinstance(item, str) for item in applicable)
    ):
        raise ConfigurationError("Select at least one valid work package.")
    unknown_packages = sorted(set(applicable) - set(packages))
    if unknown_packages:
        raise ConfigurationError(f"Unknown work packages: {unknown_packages}")
    if "terrace" in applicable and "external" not in applicable:
        raise ConfigurationError(
            "Terrace waterproofing requires the external service-duct and facade "
            "clearance package; include the external work package."
        )
    tower_ids = config.get("tower_ids")
    if tower_ids is None:
        tower_ids = [f"T{i}" for i in range(1, tower_count + 1)]
    if (
        not isinstance(tower_ids, list)
        or len(tower_ids) != tower_count
        or any(not isinstance(item, str) or not item.strip() for item in tower_ids)
        or len(set(tower_ids)) != len(tower_ids)
    ):
        raise ConfigurationError("tower_ids must contain one unique, non-empty ID per tower.")

    optional_codes = config.get("optional_activities", [])
    known_codes = {
        step["code"]
        for package in packages.values()
        for step in package.get("steps", [])
    }
    unknown_codes = sorted(set(optional_codes) - known_codes)
    if unknown_codes:
        raise ConfigurationError(f"Unknown optional activity codes: {unknown_codes}")
    return start_date, tower_ids, floors, units, staircases


def generate_schedule(
    config: dict[str, Any] | None = None,
    template_path: str | Path = DEFAULT_TEMPLATE,
) -> pd.DataFrame:
    """Expand project, tower, floor, and unit workflows into unique activity rows.

    Every default duration is explicitly marked as an illustrative assumption.
    Actual dates and actual progress are intentionally left blank.
    """
    project_config = dict(config or load_project_config())
    templates = load_templates(template_path)
    packages: dict[str, Any] = templates["packages"]
    start_date, tower_ids, floors, units_per_floor, staircase_count = _validate_config(
        project_config, packages
    )
    enabled = set(project_config.get("applicable_work_packages", packages))
    optional = set(project_config.get("optional_activities", []))
    rows: list[dict[str, Any]] = []
    generated_ids: set[str] = set()

    def emit(
        package_key: str,
        activity_prefix: str,
        tower_id: str = "SITE",
        floor_id: str = "",
        unit_id: str = "",
        start_after: list[str] | None = None,
        parent_id: str = "",
    ) -> dict[str, str]:
        if package_key not in enabled:
            return {}
        package = packages[package_key]
        previous: list[str] = list(start_after or [])
        emitted: dict[str, str] = {}
        for step in package["steps"]:
            code = step["code"]
            if step.get("conditional") and code not in optional:
                continue
            activity_id = f"{activity_prefix}-{code}" if activity_prefix else code
            if activity_id in generated_ids:
                raise ConfigurationError(f"Generated duplicate activity ID: {activity_id}")
            generated_ids.add(activity_id)
            rows.append(
                {
                    "activity_id": activity_id,
                    "activity_name": step["name"],
                    "parent_activity_id": parent_id or f"WP-{activity_prefix or 'PROJECT'}",
                    "phase": package["phase"],
                    "work_package": package["work_package"],
                    "tower_id": tower_id,
                    "floor_id": floor_id,
                    "unit_id": unit_id,
                    "trade": step["trade"],
                    "predecessors": list(previous),
                    "successors": [],
                    "duration_days": int(step["days"]),
                    "baseline_start": start_date,
                    "baseline_finish": "",
                    "actual_start": "",
                    "actual_finish": "",
                    "planned_progress_pct": None,
                    "actual_progress_pct": None,
                    "responsible_team": step["trade"],
                    "risk_level": step["risk"],
                    "activity_type": "task",
                    "is_milestone": False,
                    "is_conditional": bool(step.get("conditional", False)),
                    "applicability_condition": step.get("condition", ""),
                    "source_document_reference": (
                        f"{templates.get('source_document', 'Source sequence')} §{step['ref']}"
                    ),
                    "assumption_flag": "Illustrative duration assumption",
                    "data_classification": "Synthetic demonstration",
                }
            )
            emitted[code] = activity_id
            previous = [activity_id]
        return emitted

    site = emit("site_setup", "SITE", parent_id="WP-PROJECT-SITE-SETUP")
    site_finish = list(site.values())[-1:] if site else []
    shell_finishes: dict[str, list[str]] = {}
    block_finishes: dict[tuple[str, int], list[str]] = {}
    lobby_finishes: dict[str, list[str]] = {}
    staircase_finishes: dict[str, list[str]] = {}
    unit_finishes: dict[str, list[str]] = {}
    top_shells: dict[str, list[str]] = {}

    for tower_id in tower_ids:
        substructure = emit(
            "substructure",
            f"{tower_id}-SUB",
            tower_id=tower_id,
            start_after=site_finish,
            parent_id=f"WP-{tower_id}-SUBSTRUCTURE",
        )
        plinth_finish = substructure.get("WET-AREA-WP") or next(
            reversed(substructure.values()), None
        )
        previous_shell = [plinth_finish] if plinth_finish else []
        tower_shell_end: list[str] = []
        lift_handover = emit(
            "lift_handover",
            f"{tower_id}-LIFT",
            tower_id=tower_id,
            start_after=(
                [f"{tower_id}-F{floors:02d}-SHELL-POST-POUR-CLEAR"]
                if "floor_shell" in enabled
                else previous_shell
            ),
            parent_id=f"WP-{tower_id}-LIFT-HANDOVER",
        )
        for floor_number in range(1, floors + 1):
            floor_id = f"F{floor_number:02d}"
            floor_prefix = f"{tower_id}-{floor_id}"
            room_door_ids: list[str] = []
            shell = emit(
                "floor_shell",
                f"{floor_prefix}-SHELL",
                tower_id=tower_id,
                floor_id=floor_id,
                start_after=previous_shell,
                parent_id=f"WP-{tower_id}-{floor_id}-SHELL",
            )
            shell_clear = shell.get("POST-POUR-CLEAR")
            shell_end_id = shell.get("SLAB-POUR") or shell_clear
            if shell_end_id:
                previous_shell = [shell_end_id]
                tower_shell_end = [shell_clear or shell_end_id]
            floor_anchor = [shell_clear] if shell_clear else previous_shell

            blockwork = emit(
                "floor_blockwork",
                f"{floor_prefix}-BLOCK",
                tower_id=tower_id,
                floor_id=floor_id,
                start_after=floor_anchor,
                parent_id=f"WP-{tower_id}-{floor_id}-BLOCKWORK",
            )
            block_finish = list(blockwork.values())[-1:] or floor_anchor
            block_finishes[(tower_id, floor_number)] = block_finish

            for unit_number in range(1, units_per_floor + 1):
                unit_id = f"U{unit_number:02d}"
                unit_prefix = f"{floor_prefix}-{unit_id}"
                wet = emit(
                    "wet_area",
                    f"{unit_prefix}-WET",
                    tower_id=tower_id,
                    floor_id=floor_id,
                    unit_id=unit_id,
                    start_after=block_finish,
                    parent_id=f"WP-{tower_id}-{floor_id}-{unit_id}-WET",
                )
                room = emit(
                    "room_finish",
                    f"{unit_prefix}-ROOM",
                    tower_id=tower_id,
                    floor_id=floor_id,
                    unit_id=unit_id,
                    start_after=block_finish,
                    parent_id=f"WP-{tower_id}-{floor_id}-{unit_id}-ROOM",
                )
                wet_waterproof_id = wet.get("WET-WATERPROOF")
                room_floor_id = room.get("ROOM-FLOORING")
                if wet_waterproof_id and room_floor_id:
                    room_floor_idx = next(
                        idx
                        for idx, activity in enumerate(rows)
                        if activity["activity_id"] == room_floor_id
                    )
                    rows[room_floor_idx]["predecessors"] = list(
                        dict.fromkeys(
                            [
                                *rows[room_floor_idx]["predecessors"],
                                wet_waterproof_id,
                            ]
                        )
                    )
                if "PREHUNG-DOORS" in room:
                    room_door_ids.append(room["PREHUNG-DOORS"])
                unit_finishes[unit_prefix] = (
                    [list(wet.values())[-1]] if wet else []
                ) + ([list(room.values())[-1]] if room else [])
            lobby_start = [*floor_anchor, *list(lift_handover.values())[-1:]]
            lobby = emit(
                "lift_lobby",
                f"{floor_prefix}-LOBBY",
                tower_id=tower_id,
                floor_id=floor_id,
                start_after=lobby_start,
                parent_id=f"WP-{tower_id}-{floor_id}-LOBBY",
            )
            lobby_floor_id = lobby.get("LOBBY-FLOOR")
            if lobby_floor_id:
                for door_id in room_door_ids:
                    door_idx = next(
                        idx
                        for idx, activity in enumerate(rows)
                        if activity["activity_id"] == door_id
                    )
                    rows[door_idx]["predecessors"] = list(
                        dict.fromkeys(
                            [*rows[door_idx]["predecessors"], lobby_floor_id]
                        )
                    )
            lobby_finishes.setdefault(tower_id, []).extend(
                list(lobby.values())[-1:] or lobby_start
            )

        if "terrace" in enabled:
            external_refs: list[str] = []
            # Terrace work may precede or follow external paint; the required
            # duct and facade clearances are linked explicitly.
            external_refs.extend(
                [
                    f"EXT-SERVICE-DUCT-CLEAR",
                    f"EXT-FACADE-FRAME",
                ]
            )
            terrace_start = [*tower_shell_end, *external_refs]
            terrace = emit(
                "terrace",
                f"{tower_id}-TERRACE",
                tower_id=tower_id,
                start_after=terrace_start,
                parent_id=f"WP-{tower_id}-TERRACE",
            )
        else:
            terrace = {}

        regular_stair_finishes = []
        for stair_number in range(1, staircase_count):
            regular_id = f"{tower_id}-STAIR-S{stair_number:02d}-BOTTOM-UP-FINISH"
            rows.append(
                {
                    "activity_id": regular_id,
                    "activity_name": f"Staircase {stair_number} flooring & finishing - bottom-to-top",
                    "parent_activity_id": f"WP-{tower_id}-STAIRCASE",
                    "phase": "Common areas",
                    "work_package": "Staircase works",
                    "tower_id": tower_id,
                    "floor_id": "",
                    "unit_id": "",
                    "trade": "Finishes",
                    "predecessors": tower_shell_end,
                    "successors": [],
                    "duration_days": 5,
                    "baseline_start": start_date,
                    "baseline_finish": "",
                    "actual_start": "",
                    "actual_finish": "",
                    "planned_progress_pct": None,
                    "actual_progress_pct": None,
                    "responsible_team": "Finishes",
                    "risk_level": "High",
                    "activity_type": "task",
                    "is_milestone": False,
                    "is_conditional": False,
                    "applicability_condition": "",
                    "source_document_reference": (
                        f"{templates.get('source_document', 'Source sequence')} §D.11"
                    ),
                    "assumption_flag": "Illustrative duration assumption",
                    "data_classification": "Synthetic demonstration",
                }
            )
            regular_stair_finishes.append(regular_id)

        retained_ids: list[str] = []
        if project_config.get("retain_staircase_for_material_shifting", True):
            retained_id = (
                f"{tower_id}-STAIR-S{staircase_count:02d}-RETAINED-FINISH"
            )
            rows.append(
                {
                    "activity_id": retained_id,
                    "activity_name": "Retained staircase flooring & finishing - top-to-bottom after common-area finishing",
                    "parent_activity_id": f"WP-{tower_id}-STAIRCASE",
                    "phase": "Common areas",
                    "work_package": "Staircase works - material-shifting constraint",
                    "tower_id": tower_id,
                    "floor_id": "",
                    "unit_id": "",
                    "trade": "Finishes",
                    "predecessors": [
                        *lobby_finishes.get(tower_id, []),
                    ],
                    "successors": [],
                    "duration_days": 5,
                    "baseline_start": start_date,
                    "baseline_finish": "",
                    "actual_start": "",
                    "actual_finish": "",
                    "planned_progress_pct": None,
                    "actual_progress_pct": None,
                    "responsible_team": "Finishes",
                    "risk_level": "High",
                    "activity_type": "task",
                    "is_milestone": False,
                    "is_conditional": False,
                    "applicability_condition": "",
                    "source_document_reference": (
                        f"{templates.get('source_document', 'Source sequence')} §D.11"
                    ),
                    "assumption_flag": "Illustrative duration assumption",
                    "data_classification": "Synthetic demonstration",
                }
            )
            retained_ids = [retained_id]
        staircase_finishes[tower_id] = [*regular_stair_finishes, *retained_ids]

        cleaning = emit(
            "lobby_cleaning",
            f"{tower_id}-CLEAN",
            tower_id=tower_id,
            start_after=[
                *lobby_finishes.get(tower_id, []),
                *staircase_finishes[tower_id],
            ],
            parent_id=f"WP-{tower_id}-HANDOVER",
        )
        tower_clean = list(cleaning.values())[-1:]
        for floor_number in range(1, floors + 1):
            floor_id = f"F{floor_number:02d}"
            for unit_number in range(1, units_per_floor + 1):
                unit_id = f"U{unit_number:02d}"
                unit_prefix = f"{tower_id}-{floor_id}-{unit_id}"
                final_unit = unit_finishes.get(unit_prefix, [])
                clean = emit(
                    "unit_cleaning",
                    f"{unit_prefix}-CLEAN",
                    tower_id=tower_id,
                    floor_id=floor_id,
                    unit_id=unit_id,
                    start_after=final_unit,
                    parent_id=f"WP-{unit_prefix}-HANDOVER",
                )
                if clean:
                    unit_finishes[unit_prefix] = list(clean.values())
        shell_finishes[tower_id] = tower_shell_end

    if "external" in enabled:
        external_start = [activity_id for values in shell_finishes.values() for activity_id in values]
        external = emit(
            "external",
            "EXT",
            start_after=external_start,
            parent_id="WP-PROJECT-EXTERNAL",
        )
    else:
        external = {}

    cleaning_end_ids = [
        row["activity_id"]
        for row in rows
        if row["activity_id"].endswith("-UNIT-DEEP-CLEAN")
        or row["activity_id"].endswith("-LOBBY-DEEP-CLEAN")
    ]
    final_after = [
        *cleaning_end_ids,
        *[
            row["activity_id"]
            for row in rows
            if row["activity_id"].endswith("-TERRACE-PROTECTION")
            or row["activity_id"] == "EXT-RAINWATER-DOWNTAKE"
            or (
                row["activity_id"].startswith(tuple(tower_ids))
                and "-STAIR-" in row["activity_id"]
                and row["activity_id"].endswith("-FINISH")
            )
        ],
    ] or [
        row["activity_id"]
        for row in rows
        if row["work_package"] in {"Toilet and wet-area sequence", "Room, living and bedroom sequence"}
    ]
    final = emit(
        "final_inspection",
        "PROJECT",
        start_after=final_after,
        parent_id="WP-PROJECT-HANDOVER",
    )

    frame = pd.DataFrame(rows, columns=ACTIVITY_COLUMNS)
    if frame.empty:
        raise ConfigurationError("The selected work packages generated no activities.")
    if frame["activity_id"].duplicated().any():
        raise ConfigurationError("Generated schedule contains duplicate activity IDs.")
    known = set(frame["activity_id"])
    frame["predecessors"] = frame["predecessors"].apply(
        lambda preds: list(dict.fromkeys(pred for pred in preds if pred in known))
    )
    successors: dict[str, list[str]] = {activity_id: [] for activity_id in known}
    for row in frame.itertuples():
        for predecessor in row.predecessors:
            successors[predecessor].append(row.activity_id)
    frame["successors"] = frame["activity_id"].map(successors)

    # Validate dates and graph immediately; template edits must not silently
    # create a broken generated project.
    calculated = sch.compute_cpm(frame)
    frame["baseline_finish"] = frame["activity_id"].map(
        lambda activity_id: calculated.activities[activity_id].earliest_finish
    )
    return frame.reset_index(drop=True)


def export_activity_register(frame: pd.DataFrame, path: str | Path) -> None:
    """Export the editable activity register to CSV."""
    export = frame.copy(deep=True)
    for column in ("predecessors", "successors"):
        if column in export:
            export[column] = export[column].apply(
                lambda values: ";".join(sch.parse_predecessors(values))
            )
    export.to_csv(path, index=False, date_format="%Y-%m-%d")


def import_activity_register(upload: Any) -> pd.DataFrame:
    """Parse and validate an uploaded CSV without mutating the active schedule."""
    try:
        candidate = pd.read_csv(upload)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, TypeError) as exc:
        raise ConfigurationError(f"Could not parse the uploaded CSV: {exc}") from exc
    missing = sorted(set(ACTIVITY_COLUMNS) - set(candidate.columns))
    if missing:
        raise ConfigurationError(f"Activity register is missing columns: {missing}")
    candidate = candidate[ACTIVITY_COLUMNS].copy()
    def parse_links(value: Any) -> list[str]:
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value
            if isinstance(parsed, (list, tuple)):
                return sch.parse_predecessors(parsed)
        return sch.parse_predecessors(value)

    candidate["predecessors"] = candidate["predecessors"].apply(parse_links)
    try:
        candidate["baseline_start"] = candidate["baseline_start"].apply(
            lambda value: sch.parse_iso_date(value, "uploaded activity")
        )
    except sch.ScheduleError as exc:
        raise ConfigurationError(str(exc)) from exc
    candidate["is_milestone"] = candidate["is_milestone"].apply(
        lambda value: sch.parse_boolean(value, "is_milestone")
    )
    candidate["duration_days"] = pd.to_numeric(candidate["duration_days"], errors="coerce")
    if candidate["duration_days"].isna().any() or (
        candidate["duration_days"] < 0
    ).any() or (candidate["duration_days"] % 1 != 0).any():
        raise ConfigurationError("Durations must be whole calendar days.")
    invalid_zero = (candidate["duration_days"] == 0) & ~candidate["is_milestone"]
    if invalid_zero.any():
        raise ConfigurationError("A zero-day duration is allowed only for a milestone.")
    try:
        for column in ("actual_start", "actual_finish"):
            candidate[column] = candidate[column].apply(
                lambda value: (
                    ""
                    if pd.isna(value) or str(value).strip() == ""
                    else sch.parse_iso_date(value, f"uploaded {column}").isoformat()
                )
            )
    except sch.ScheduleError as exc:
        raise ConfigurationError(str(exc)) from exc
    dated = candidate.loc[
        (candidate["actual_start"] != "") & (candidate["actual_finish"] != "")
    ]
    if any(
        date.fromisoformat(str(row.actual_finish))
        < date.fromisoformat(str(row.actual_start))
        for row in dated.itertuples()
    ):
        raise ConfigurationError("An actual finish cannot be earlier than actual start.")
    for column in ("planned_progress_pct", "actual_progress_pct"):
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
        invalid = candidate[column].notna() & ~candidate[column].between(0, 100)
        if invalid.any():
            raise ConfigurationError(f"{column} values must be between 0 and 100.")
    try:
        sch.compute_cpm(candidate)
    except sch.ScheduleError as exc:
        raise ConfigurationError(str(exc)) from exc
    known = set(candidate["activity_id"])
    for activity_id, predecessors in zip(candidate["activity_id"], candidate["predecessors"]):
        missing_ids = sorted(set(predecessors) - known)
        if missing_ids:
            raise ConfigurationError(
                f"{activity_id} references missing predecessor IDs: {missing_ids}"
            )
    successors = {activity_id: [] for activity_id in known}
    for activity_id, predecessors in zip(
        candidate["activity_id"], candidate["predecessors"]
    ):
        for predecessor in predecessors:
            successors[predecessor].append(activity_id)
    candidate["successors"] = candidate["activity_id"].map(successors)
    return candidate
