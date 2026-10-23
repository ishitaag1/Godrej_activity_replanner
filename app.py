"""Streamlit project-controls dashboard for synthetic construction schedules."""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import activity_generator as generator
import progress_tracker
import scenario_manager
import scheduler as sch
import sequence_validator

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "data" / "demo_project_config.json"

st.set_page_config(
    page_title="Construction Schedule Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize() -> None:
    if "project_config" not in st.session_state:
        st.session_state.project_config = generator.load_project_config(CONFIG_PATH)
        st.session_state.activity_frame = generator.generate_schedule(
            st.session_state.project_config
        )
        st.session_state.baseline_frame = st.session_state.activity_frame.copy(deep=True)
        st.session_state.baseline_schedule = sch.compute_cpm(
            st.session_state.baseline_frame
        )
        st.session_state.current_schedule = st.session_state.baseline_schedule
        st.session_state.delay_scenario = None
        st.session_state.scenarios = []
        st.session_state.import_message = None
        st.session_state.unsaved_register_edits = False


def _style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7f9; color: #1f2933; }
        .block-container { max-width: 1500px; padding-top: 1.25rem; }
        h1, h2, h3 { color: #17324d; letter-spacing: -0.02em; }
        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e2e8ee;
            padding: .8rem 1rem; border-radius: .55rem;
        }
        div[data-testid="stMetricValue"] { color: #17324d; }
        .assumption-note { color: #526273; font-size: .92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _current_context() -> tuple[pd.DataFrame, sch.ProjectSchedule]:
    frame: pd.DataFrame = st.session_state.activity_frame
    schedule = sch.compute_cpm(frame)
    st.session_state.current_schedule = schedule
    return frame, schedule


def _register_table(
    frame: pd.DataFrame, schedule: sch.ProjectSchedule
) -> pd.DataFrame:
    result = frame.copy(deep=True)
    computed = schedule.to_dataframe().set_index("activity_id")
    for column in (
        "earliest_start",
        "earliest_finish",
        "latest_start",
        "latest_finish",
        "total_float_days",
        "free_float_days",
        "is_critical",
    ):
        result[column] = result["activity_id"].map(computed[column])
    result["status"] = progress_tracker.build_progress_report(
        frame, schedule, date.today()
    ).set_index("activity_id")["status"].reindex(result["activity_id"]).to_numpy()
    return result


def _render_overview(
    frame: pd.DataFrame,
    baseline: sch.ProjectSchedule,
    current: sch.ProjectSchedule,
) -> None:
    progress = progress_tracker.build_progress_report(frame, current, date.today())
    actual_available = progress["actual_progress_pct"].notna().any()
    completed = int((progress["status"] == "Completed").sum())
    in_progress = int((progress["status"] == "In progress").sum())
    not_started = int((progress["status"] == "Not started").sum())
    overdue = int(progress["overdue"].sum())
    high_risk = int((frame["risk_level"] == "High").sum())
    slip = (current.project_finish - baseline.project_finish).days

    st.subheader("Project overview")
    with st.container(horizontal=True):
        st.metric("Project start", current.project_start.isoformat(), border=True)
        st.metric("Baseline completion", baseline.project_finish.isoformat(), border=True)
        st.metric(
            "Current forecast",
            current.project_finish.isoformat(),
            delta=slip,
            delta_description="calendar days versus baseline",
            delta_color="inverse",
            border=True,
        )
        st.metric(
            "Duration", f"{current.project_duration_days} days", border=True
        )
        st.metric("Activities", f"{len(current.activities):,}", border=True)

    with st.container(horizontal=True):
        st.metric("Completed", completed, border=True)
        st.metric("In progress", in_progress, border=True)
        st.metric("Not started", not_started, border=True)
        st.metric("Critical", len(current.critical_activities), border=True)
        st.metric("Overdue", overdue, border=True)
        st.metric("High risk", high_risk, border=True)
    st.caption(
        "Synthetic demonstration schedule. Default durations are illustrative "
        "calendar-day assumptions, not verified project durations. Actual progress "
        "remains blank until entered or imported."
    )
    if not actual_available:
        st.info("No actual progress has been entered; actual progress is not inferred from dates.")
    if current.critical_path:
        st.write("**Current critical path:** " + " → ".join(current.critical_path))

    phase_col, location_col = st.columns(2)
    with phase_col:
        phase = frame.groupby("phase", as_index=False)["activity_id"].count()
        phase = phase.rename(columns={"activity_id": "activities"})
        fig = px.bar(
            phase,
            x="activities",
            y="phase",
            orientation="h",
            color_discrete_sequence=["#2878a5"],
            title="Activities by construction phase",
        )
        fig.update_layout(height=330, margin=dict(l=10, r=20, t=50, b=10))
        st.plotly_chart(fig, width="stretch", alt="Activity counts by construction phase")
    with location_col:
        floor_frame = frame.loc[frame["floor_id"] != ""]
        floor_counts = (
            floor_frame.groupby(["tower_id", "floor_id"], as_index=False)["activity_id"]
            .count()
            .rename(columns={"activity_id": "activities"})
        )
        floor_counts["location"] = (
            floor_counts["tower_id"] + " · " + floor_counts["floor_id"]
        )
        fig = px.bar(
            floor_counts,
            x="activities",
            y="location",
            orientation="h",
            title="Activities by tower and floor",
            color_discrete_sequence=["#4c9476"],
        )
        fig.update_layout(height=330, margin=dict(l=10, r=20, t=50, b=10))
        st.plotly_chart(fig, width="stretch", alt="Activity counts by tower and floor")

    risk_col, critical_col = st.columns(2)
    with risk_col:
        risks = frame.groupby("risk_level", as_index=False)["activity_id"].count()
        risks = risks.rename(columns={"activity_id": "activities"})
        fig = px.pie(
            risks,
            values="activities",
            names="risk_level",
            hole=0.55,
            color="risk_level",
            color_discrete_map={
                "High": "#bd4b4b",
                "Medium": "#d69a32",
                "Low": "#4c9476",
            },
            title="Illustrative activity risk distribution",
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", alt="Distribution of illustrative activity risk levels")
    with critical_col:
        critical_counts = pd.DataFrame(
            {
                "criticality": ["Critical", "Non-critical"],
                "activities": [
                    len(current.critical_activities),
                    len(current.activities) - len(current.critical_activities),
                ],
            }
        )
        fig = px.pie(
            critical_counts,
            values="activities",
            names="criticality",
            hole=0.55,
            color="criticality",
            color_discrete_map={"Critical": "#bf4c4c", "Non-critical": "#2878a5"},
            title="Critical versus non-critical activities",
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch", alt="CPM critical and non-critical activity counts")

    if actual_available:
        chart = progress.dropna(subset=["actual_progress_pct"])
        chart = chart.groupby("tower_id", as_index=False).agg(
            actual=("actual_progress_pct", "mean"),
            planned=("planned_progress_pct", "mean"),
        )
        st.bar_chart(
            chart.set_index("tower_id")[["planned", "actual"]],
            alt="Mean planned and actual progress by tower",
        )
    else:
        st.caption("Planned-versus-actual chart will appear after progress data is supplied.")


def _render_register(
    frame: pd.DataFrame, schedule: sch.ProjectSchedule
) -> None:
    table = _register_table(frame, schedule)
    row_a, row_b, row_c, row_d = st.columns([2, 1, 1, 1])
    query = row_a.text_input("Search activity ID or name", key="register_search")
    towers = ["All", *sorted(table["tower_id"].unique())]
    tower = row_b.selectbox("Tower", towers, key="register_tower")
    phases = ["All", *sorted(table["phase"].unique())]
    phase = row_c.selectbox("Phase", phases, key="register_phase")
    critical_only = row_d.checkbox("Critical only", key="register_critical")
    row_a2, row_b2, row_c2, row_d2 = st.columns(4)
    floors = ["All", *sorted(set(table["floor_id"]) - {""})]
    floor = row_a2.selectbox("Floor", floors, key="register_floor")
    trades = ["All", *sorted(table["trade"].fillna("").unique())]
    trade = row_b2.selectbox("Trade", trades, key="register_trade")
    risk = row_c2.selectbox(
        "Risk", ["All", *sorted(table["risk_level"].fillna("").unique())],
        key="register_risk",
    )
    status = row_d2.selectbox(
        "Status", ["All", "Completed", "In progress", "Not started", "Overdue"],
        key="register_status",
    )
    result = table.copy()
    if tower != "All":
        result = result[result["tower_id"] == tower]
    if floor != "All":
        result = result[result["floor_id"] == floor]
    if phase != "All":
        result = result[result["phase"] == phase]
    if trade != "All":
        result = result[result["trade"] == trade]
    if risk != "All":
        result = result[result["risk_level"] == risk]
    if status != "All":
        result = result[result["status"] == status]
    if critical_only:
        result = result[result["is_critical"]]
    if query.strip():
        needle = query.strip().casefold()
        result = result[
            result["activity_id"].str.casefold().str.contains(needle, regex=False)
            | result["activity_name"].str.casefold().str.contains(needle, regex=False)
        ]

    st.caption(f"{len(result):,} matching activities.")
    show_all = st.checkbox("Show all matching rows for editing", value=False)
    if not show_all and len(result) > 200:
        st.info("Showing the first 200 matches. Filter further or enable all rows to edit the complete selection.")
        result = result.head(200)
    result = result.copy()
    result["predecessors"] = result["predecessors"].apply(
        lambda values: ";".join(sch.parse_predecessors(values))
    )
    register_view = st.segmented_control(
        "Register view",
        ["Schedule", "Progress"],
        default="Schedule",
        key="register_view",
    )
    if register_view == "Progress":
        editable_columns = [
            "activity_id",
            "activity_name",
            "tower_id",
            "floor_id",
            "planned_progress_pct",
            "actual_start",
            "actual_finish",
            "actual_progress_pct",
        ]
    else:
        editable_columns = [
            "activity_id",
            "activity_name",
            "tower_id",
            "floor_id",
            "duration_days",
            "is_milestone",
            "predecessors",
            "baseline_start",
            "total_float_days",
            "is_critical",
        ]
    edited = st.data_editor(
        result[editable_columns],
        width="stretch",
        alt="Editable activity register with dates, dependency links, durations and progress",
        hide_index=True,
        disabled=[
            "activity_id",
            "activity_name",
            "tower_id",
            "floor_id",
            "total_float_days",
            "is_critical",
        ],
        key="activity_register_editor",
        num_rows="fixed",
    )
    if st.button("Apply register edits and recalculate", type="primary"):
        updated = st.session_state.activity_frame.copy(deep=True)
        try:
            for _, edited_row in edited.iterrows():
                idx = updated.index[
                    updated["activity_id"] == edited_row["activity_id"]
                ][0]
                editable_fields = {
                    "duration_days",
                    "is_milestone",
                    "predecessors",
                    "baseline_start",
                    "planned_progress_pct",
                    "actual_start",
                    "actual_finish",
                    "actual_progress_pct",
                }
                for column in editable_fields.intersection(edited.columns):
                    updated.at[idx, column] = edited_row[column]
                if "predecessors" in edited.columns:
                    updated.at[idx, "predecessors"] = sch.parse_predecessors(
                        edited_row["predecessors"]
                    )
            updated["duration_days"] = pd.to_numeric(
                updated["duration_days"], errors="raise"
            )
            if (
                (updated["duration_days"] < 0).any()
                or (updated["duration_days"] % 1 != 0).any()
            ):
                raise sch.InvalidDurationError(
                    "Durations must be whole calendar days."
                )
            invalid_zero = (updated["duration_days"] == 0) & ~updated[
                "is_milestone"
            ].apply(lambda value: sch.parse_boolean(value, "is_milestone"))
            if invalid_zero.any():
                raise sch.InvalidDurationError(
                    "A zero-day duration is allowed only for a milestone."
                )
            updated["is_milestone"] = updated["is_milestone"].apply(
                lambda value: sch.parse_boolean(value, "is_milestone")
            )
            updated["baseline_start"] = updated["baseline_start"].apply(
                lambda value: sch.parse_iso_date(value, "edited activity")
            )
            for column in ("actual_start", "actual_finish"):
                updated[column] = updated[column].apply(
                    lambda value: (
                        ""
                        if pd.isna(value) or str(value).strip() == ""
                        else sch.parse_iso_date(value, f"edited {column}").isoformat()
                    )
                )
            for column in ("planned_progress_pct", "actual_progress_pct"):
                updated[column] = pd.to_numeric(updated[column], errors="raise")
                invalid = updated[column].notna() & ~updated[column].between(0, 100)
                if invalid.any():
                    raise sch.ScheduleError(f"{column} must be between 0 and 100.")
            dated = updated.loc[
                (updated["actual_start"] != "")
                & (updated["actual_finish"] != "")
            ]
            if any(
                date.fromisoformat(str(row.actual_finish))
                < date.fromisoformat(str(row.actual_start))
                for row in dated.itertuples()
            ):
                raise sch.ScheduleError(
                    "An actual finish cannot be earlier than actual start."
                )
            sch.compute_cpm(updated)
            successor_map = {
                activity_id: [] for activity_id in updated["activity_id"]
            }
            for _, activity in updated.iterrows():
                for predecessor in activity["predecessors"]:
                    successor_map[predecessor].append(activity["activity_id"])
            updated["successors"] = updated["activity_id"].map(successor_map)
        except (ValueError, TypeError, IndexError, sch.ScheduleError) as exc:
            st.error(f"Edits were not applied: {exc}")
        else:
            st.session_state.activity_frame = updated
            st.session_state.unsaved_register_edits = False
            st.success("Validated edits applied. The original generated baseline is unchanged.")
            st.rerun()

    detail_id = st.selectbox(
        "Inspect activity dependencies",
        table["activity_id"].tolist(),
        key="activity_detail_id",
    )
    item = schedule.activities[detail_id]
    st.write(
        f"**{item.activity_name}** · {item.earliest_start} → {item.earliest_finish} · "
        f"total float {item.total_float_days}d · free float {item.free_float_days}d"
    )
    st.write("**Predecessors:** " + (", ".join(item.predecessors) or "None"))
    row = frame.loc[frame["activity_id"] == detail_id].iloc[0]
    st.write("**Successors:** " + (", ".join(row["successors"]) or "None"))
    st.caption(f"Source: {row['source_document_reference']} · {row['assumption_flag']}")


def _gantt_figure(
    frame: pd.DataFrame,
    baseline: sch.ProjectSchedule,
    current: sch.ProjectSchedule,
    scenario: sch.ProjectSchedule | None,
    tower: str,
    floor: str,
    phase: str,
) -> go.Figure:
    selected = frame.copy()
    if tower != "All":
        selected = selected[selected["tower_id"] == tower]
    if floor != "All":
        selected = selected[selected["floor_id"] == floor]
    if phase != "All":
        selected = selected[selected["phase"] == phase]
    if len(selected) > 500:
        selected = selected.head(500)
    rows = []
    schedules = [("Current forecast", current)]
    if scenario is not None:
        schedules = [("Baseline", baseline), ("Scenario", scenario)]
    for _, activity in selected.iterrows():
        activity_id = activity["activity_id"]
        for label, schedule in schedules:
            item = schedule.activities[activity_id]
            rows.append(
                {
                    "activity_id": activity_id,
                    "label": (
                        f"{activity_id} · {label}"
                        if scenario is not None
                        else activity_id
                    ),
                    "series": label,
                    "start": item.earliest_start,
                    "finish_exclusive": item.earliest_finish + pd.Timedelta(days=1),
                    "criticality": "Critical" if item.is_critical else "Non-critical",
                    "duration_days": item.duration_days,
                    "total_float": item.total_float_days,
                    "progress_pct": activity["actual_progress_pct"],
                    "risk": activity["risk_level"],
                    "predecessors": ", ".join(item.predecessors),
                    "activity_name": item.activity_name,
                    "tower_id": item.tower_id,
                    "floor_id": activity["floor_id"],
                    "is_milestone": item.is_milestone,
                }
            )
    gantt = pd.DataFrame(rows)
    if gantt.empty:
        return go.Figure().update_layout(title="No activities match the selected filters.")
    task_rows = gantt.loc[~gantt["is_milestone"]]
    if task_rows.empty:
        fig = go.Figure()
    else:
        fig = px.timeline(
            task_rows,
            x_start="start",
            x_end="finish_exclusive",
            y="label",
            color="criticality",
            color_discrete_map={"Critical": "#bf4c4c", "Non-critical": "#2878a5"},
            hover_data=[
                "activity_id",
                "activity_name",
                "tower_id",
                "floor_id",
                "duration_days",
                "total_float",
                "progress_pct",
                "risk",
                "predecessors",
            ],
        )
    milestones = gantt.loc[gantt["is_milestone"]]
    for (series_name, criticality), points in milestones.groupby(
        ["series", "criticality"], sort=True
    ):
        fig.add_trace(
            go.Scatter(
                x=points["start"],
                y=points["label"],
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 11,
                    "color": "#bf4c4c" if criticality == "Critical" else "#2878a5",
                },
                name=(
                    f"Milestone · {series_name}"
                    if scenario is not None
                    else "Milestone"
                ),
                customdata=points[
                    [
                        "activity_id",
                        "activity_name",
                        "duration_days",
                        "total_float",
                        "progress_pct",
                        "risk",
                        "predecessors",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Duration: %{customdata[2]}d · Float: %{customdata[3]}d<br>"
                    "Progress: %{customdata[4]} · Risk: %{customdata[5]}<br>"
                    "Predecessors: %{customdata[6]}<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(autorange="reversed", showticklabels=len(selected) <= 70)
    fig.update_layout(
        height=min(900, max(430, 24 * min(len(selected), 28))),
        legend_title_text="Criticality",
        margin=dict(l=10, r=20, t=50, b=10),
        xaxis_title="Calendar date (inclusive finish shown through next-day boundary)",
        yaxis_title="Activity ID",
    )
    return fig


def _network_figure(
    frame: pd.DataFrame, schedule: sch.ProjectSchedule, focus_id: str
) -> go.Figure:
    graph = sch.build_graph(frame)
    connected = nx.ancestors(graph, focus_id) | nx.descendants(graph, focus_id) | {focus_id}
    if len(connected) > 120:
        upstream_distances = nx.single_source_shortest_path_length(
            graph.reverse(copy=False), focus_id
        )
        downstream_distances = nx.single_source_shortest_path_length(graph, focus_id)
        upstream = sorted(
            (item for item in upstream_distances if item != focus_id),
            key=lambda item: (upstream_distances[item], item),
        )[:59]
        downstream = sorted(
            (item for item in downstream_distances if item != focus_id),
            key=lambda item: (downstream_distances[item], item),
        )[:59]
        connected = set(upstream + downstream + [focus_id])
    subgraph = graph.subgraph(connected)
    layers = list(nx.topological_generations(subgraph))
    positions: dict[str, tuple[int, int]] = {}
    for x, layer in enumerate(layers):
        ordered = sorted(layer)
        for y, activity_id in enumerate(ordered):
            positions[activity_id] = (x, y)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in subgraph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_ids = list(positions)
    colors = [
        "#bf4c4c" if schedule.activities[item].is_critical else "#2878a5"
        for item in node_ids
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#bdc7d0", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[item][0] for item in node_ids],
            y=[positions[item][1] for item in node_ids],
            mode="markers+text",
            text=(
                node_ids
                if len(node_ids) <= 50
                else [item if item == focus_id else "" for item in node_ids]
            ),
            textposition="top center",
            marker={"size": 12, "color": colors},
            hovertext=[
                f"{item}: {schedule.activities[item].activity_name}<br>"
                f"{schedule.activities[item].earliest_start} → "
                f"{schedule.activities[item].earliest_finish}<br>"
                f"Float: {schedule.activities[item].total_float_days}d"
                for item in node_ids
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Dependency focus: {focus_id} and its connected activities",
        height=580,
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="#ffffff",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def _render_delay_analysis(
    frame: pd.DataFrame,
    current: sch.ProjectSchedule,
) -> None:
    st.subheader("Simultaneous delay events")
    activity_ids = sorted(current.activities)
    selected = st.multiselect(
        "Activities receiving a delay",
        activity_ids,
        help="All selected delays are applied together and the network is recalculated once.",
    )
    events: dict[str, int] = {}
    for activity_id in selected:
        item = current.activities[activity_id]
        value = st.number_input(
            f"{activity_id} · {item.activity_name} (delay days)",
            min_value=0,
            max_value=365,
            step=1,
            value=1,
            key=f"delay_{activity_id}",
        )
        events[activity_id] = int(value)
    if st.button("Run combined scenario", type="primary"):
        try:
            revised_frame, _, revised, impact = sch.apply_delays(frame, events)
        except sch.ScheduleError as exc:
            st.error(str(exc))
        else:
            st.session_state.delay_scenario = {
                "frame": revised_frame,
                "schedule": revised,
                "impact": impact,
                "events": events,
            }
            st.success("Scenario recalculated from the current forecast.")

    scenario = st.session_state.delay_scenario
    if not scenario:
        st.info("Select one or more activities and run a scenario to see delay propagation.")
        return
    impact = scenario["impact"]
    cols = st.columns(3)
    cols[0].metric("Current completion", impact["original_project_finish"])
    cols[1].metric(
        "Revised completion",
        impact["revised_project_finish"],
        delta=f"{impact['project_finish_slip_days']:+} calendar days",
    )
    cols[2].metric("Affected activities", impact["affected_activity_count"])
    if impact["project_finish_slip_days"] == 0:
        st.success("The completion date is unchanged; the network's remaining float absorbed this scenario.")
    else:
        st.warning(
            f"Project completion slips {impact['project_finish_slip_days']} day(s). "
            "The combined duration extensions exceeded the available float on the "
            "controlling path."
        )
    st.write("**Delay events:** " + ", ".join(
        f"{activity_id}: +{days}d" for activity_id, days in scenario["events"].items()
    ))
    st.write("**Original critical path:** " + " → ".join(impact["original_critical_path"]))
    st.write("**Revised critical path:** " + " → ".join(impact["revised_critical_path"]))
    st.write(
        "**Newly critical activities:** "
        + (", ".join(impact["newly_critical_activities"]) or "None")
    )
    affected = pd.DataFrame(impact["affected_activities"])
    if not affected.empty:
        st.dataframe(
            affected[
                [
                    "activity_id",
                    "activity_name",
                    "original_start",
                    "revised_start",
                    "start_slip_days",
                    "original_finish",
                    "revised_finish",
                    "finish_slip_days",
                    "original_float_days",
                    "revised_float_days",
                    "float_consumed_days",
                    "is_critical",
                ]
            ],
            width="stretch",
            alt="Activities affected by the combined delay scenario",
            hide_index=True,
        )
    name = st.text_input("Save this scenario as", value="Scenario A")
    if st.button("Save scenario for comparison"):
        try:
            saved = scenario_manager.calculate_scenario(
                frame, name, scenario["events"]
            )
        except sch.ScheduleError as exc:
            st.error(str(exc))
        else:
            st.session_state.scenarios = [
                item for item in st.session_state.scenarios if item.name != saved.name
            ] + [saved]
            st.success(f"Saved {saved.name}.")


def _render_progress(frame: pd.DataFrame, schedule: sch.ProjectSchedule) -> None:
    as_of = st.date_input("Status date", value=date.today(), key="progress_as_of")
    try:
        report = progress_tracker.build_progress_report(frame, schedule, as_of)
    except sch.ScheduleError as exc:
        st.error(str(exc))
        return
    if report["actual_progress_pct"].notna().any():
        st.success("Actual progress is available from explicitly entered/imported values.")
    else:
        st.info("Actual progress is not available. The dashboard does not infer it from dates.")
    cols = st.columns(4)
    cols[0].metric("Completed", int((report["status"] == "Completed").sum()))
    cols[1].metric("In progress", int((report["status"] == "In progress").sum()))
    cols[2].metric("Not started", int((report["status"] == "Not started").sum()))
    cols[3].metric("Overdue", int(report["overdue"].sum()))
    st.dataframe(
        report[
            [
                "activity_id",
                "activity_name",
                "planned_progress_pct",
                "actual_progress_pct",
                "actual_start",
                "actual_finish",
                "remaining_duration_days",
                "forecast_finish",
                "status",
                "at_risk",
                "schedule_variance_days",
            ]
        ],
        width="stretch",
        alt="Progress, remaining duration, forecast finish and overdue status by activity",
        hide_index=True,
        height=520,
    )


def _render_sequence_validation(frame: pd.DataFrame) -> None:
    violations = sequence_validator.validate_construction_sequence(frame)
    if not violations:
        st.success("No encoded mandatory sequence violations were found.")
        st.caption(
            "The validator checks only documented relationships represented by "
            "recognized activity IDs; it does not silently change dependencies."
        )
        return
    st.warning(f"{len(violations)} sequence item(s) need review.")
    st.dataframe(
        pd.DataFrame(violations),
        width="stretch",
        alt="Construction sequence validation findings and recommended actions",
        hide_index=True,
    )


def _render_import_export(frame: pd.DataFrame) -> None:
    st.subheader("Activity register import / export")
    uploaded = st.file_uploader("Upload an activity register CSV", type=["csv"])
    if uploaded is not None and st.button("Validate and apply uploaded CSV"):
        try:
            imported = generator.import_activity_register(uploaded)
        except (generator.ConfigurationError, sch.ScheduleError) as exc:
            st.error(str(exc))
        else:
            st.session_state.activity_frame = imported
            st.session_state.delay_scenario = None
            st.success(f"Applied validated register with {len(imported):,} activities.")
            st.rerun()
    csv_export = frame.copy(deep=True)
    for column in ("predecessors", "successors"):
        csv_export[column] = csv_export[column].apply(
            lambda values: ";".join(sch.parse_predecessors(values))
        )
    csv_data = csv_export.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    st.download_button(
        "Download current activity register",
        data=csv_data,
        file_name="construction_activity_register.csv",
        mime="text/csv",
    )
    json_data = frame.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")
    st.download_button(
        "Download current activity register JSON",
        data=json_data,
        file_name="construction_activity_register.json",
        mime="application/json",
    )
    scenario_items = st.session_state.scenarios
    if scenario_items:
        export_type = st.selectbox("Scenario export format", ["json", "csv"])
        content = scenario_manager.export_scenarios(
            st.session_state.baseline_schedule, scenario_items, export_type
        )
        st.download_button(
            "Download scenario comparison",
            content,
            file_name=f"scenario_comparison.{export_type}",
            mime="application/json" if export_type == "json" else "text/csv",
        )
    st.caption(
        "Imported registers are validated for required columns, IDs, dates, "
        "whole-day durations, predecessors, and cycles before replacing the active data."
    )


def _render_assistant(frame: pd.DataFrame) -> None:
    try:
        from agent import (
            AgentError,
            AgentUnavailableError,
            ScheduleAssistant,
            SETUP_INSTRUCTIONS,
            check_ollama_status,
        )
    except ImportError as exc:
        st.error(f"Optional assistant module could not be loaded: {exc}")
        return
    status = check_ollama_status()
    if not status["model_ready"]:
        st.info("The optional Ollama assistant is unavailable; all deterministic project controls remain usable.")
        st.markdown(SETUP_INSTRUCTIONS.replace("\n", "  \n"))
        if status.get("detail"):
            st.caption(status["detail"])
        return
    st.success(status["message"])
    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []
    for message in st.session_state.assistant_history:
        with st.chat_message(message["role"]):
            st.write(message["text"])
    prompt = st.chat_input("Ask about completion, criticality, dependencies or a delay")
    if not prompt:
        return
    assistant = ScheduleAssistant(frame=frame)
    with st.spinner("Consulting deterministic schedule tools..."):
        try:
            answer = assistant.ask(prompt)
        except (AgentUnavailableError, AgentError) as exc:
            st.error(str(exc))
            return
    st.session_state.assistant_history.extend(
        [
            {"role": "user", "text": prompt},
            {"role": "assistant", "text": answer["answer"]},
        ]
    )
    st.write(answer["answer"])
    st.caption(
        "Tools executed: "
        + (", ".join(item["name"] for item in answer["tools_used"]) or "none")
        + (" · scheduler-grounded" if answer["grounded"] else " · not grounded")
    )


def _render_settings() -> None:
    config = st.session_state.project_config
    st.sidebar.header("Project configuration")
    with st.sidebar.form("project_configuration"):
        start = st.date_input(
            "Project start date",
            value=date.fromisoformat(config["project_start_date"]),
        )
        tower_count = st.number_input(
            "Towers / wings", 1, 20, int(config["tower_count"])
        )
        floors = st.number_input(
            "Floors per tower", 1, 200, int(config["floors_per_tower"])
        )
        units = st.number_input(
            "Units per floor", 1, 100, int(config.get("units_per_floor", 1))
        )
        staircase_count = st.number_input(
            "Staircases per tower", 1, 20,
            int(config.get("staircase_count_per_tower", 2)),
        )
        packages = st.multiselect(
            "Applicable work packages",
            list(generator.load_templates()["packages"]),
            default=config["applicable_work_packages"],
        )
        conditional_codes = sorted(
            step["code"]
            for package in generator.load_templates()["packages"].values()
            for step in package["steps"]
            if step.get("conditional")
        )
        optional_activities = st.multiselect(
            "Activate conditional activities",
            conditional_codes,
            default=config.get("optional_activities", []),
        )
        retain_stair = st.checkbox(
            "Retain staircase for material shifting",
            value=bool(config.get("retain_staircase_for_material_shifting", True)),
        )
        discard = st.checkbox(
            "I understand regeneration replaces current register edits and scenarios",
            value=False,
        )
        regenerate = st.form_submit_button("Regenerate schedule")
    if regenerate:
        updated_config = {
            **config,
            "project_start_date": start.isoformat(),
            "tower_count": int(tower_count),
            "floors_per_tower": int(floors),
            "units_per_floor": int(units),
            "staircase_count_per_tower": int(staircase_count),
            "applicable_work_packages": packages,
            "optional_activities": optional_activities,
            "retain_staircase_for_material_shifting": retain_stair,
        }
        if not discard:
            st.sidebar.error("Confirm the reset warning before regenerating.")
        else:
            try:
                generated = generator.generate_schedule(updated_config)
                schedule = sch.compute_cpm(generated)
            except (generator.ConfigurationError, sch.ScheduleError) as exc:
                st.sidebar.error(str(exc))
            else:
                st.session_state.project_config = updated_config
                st.session_state.activity_frame = generated
                st.session_state.baseline_frame = generated.copy(deep=True)
                st.session_state.baseline_schedule = schedule
                st.session_state.current_schedule = schedule
                st.session_state.delay_scenario = None
                st.session_state.scenarios = []
                st.sidebar.success(f"Generated {len(generated):,} activities.")
                st.rerun()
    st.sidebar.caption(
        "Calendar convention: whole calendar days; inclusive finish date; "
        "successor starts the next day. No resource leveling."
    )


def main() -> None:
    _initialize()
    _style()
    _render_settings()
    frame, current = _current_context()
    baseline = st.session_state.baseline_schedule
    scenario = st.session_state.delay_scenario

    st.title("Construction Schedule Intelligence")
    st.caption(
        "Construction programme, critical-path monitoring and delay-impact analysis · "
        "synthetic demonstration data"
    )
    tabs = st.tabs(
        [
            "Project Overview",
            "Activity Register",
            "Gantt & Timeline",
            "Critical Path & Dependencies",
            "Delay Impact Analysis",
            "Progress & Forecast",
            "Sequence Validation",
            "Scenario Comparison",
            "Data Import / Export",
            "AI Schedule Assistant",
        ],
        on_change="rerun",
    )

    def render_gantt() -> None:
        cols = st.columns(3)
        towers = ["All", *sorted(set(frame["tower_id"]) - {"SITE"})]
        tower = cols[0].selectbox("Tower", towers, key="gantt_tower")
        floors = ["All", *sorted(set(frame["floor_id"]) - {""})]
        floor = cols[1].selectbox("Floor", floors, key="gantt_floor")
        phases = ["All", *sorted(frame["phase"].unique())]
        phase = cols[2].selectbox("Phase", phases, key="gantt_phase")
        scenario_schedule = scenario["schedule"] if scenario else None
        st.plotly_chart(
            _gantt_figure(frame, baseline, current, scenario_schedule, tower, floor, phase),
            width="stretch",
            alt="CPM-computed activity timeline by selected tower, floor and phase",
        )

    def render_critical_path() -> None:
        focus_id = st.selectbox(
            "Focus activity", sorted(current.activities), key="network_focus"
        )
        selected = current.activities[focus_id]
        st.write(
            f"**{selected.activity_name}** · total float {selected.total_float_days}d · "
            f"free float {selected.free_float_days}d · "
            f"{'critical' if selected.is_critical else 'non-critical'}"
        )
        st.caption("Red nodes are CPM-critical. The view is limited to connected activities around the selection.")
        st.plotly_chart(
            _network_figure(frame, current, focus_id),
            width="stretch",
            alt="Upstream and downstream dependency network around the selected activity",
        )
        st.write("**Critical paths**")
        for path in current.critical_paths[:20]:
            st.code(" → ".join(path))
        if len(current.critical_paths) > 20:
            st.caption(f"{len(current.critical_paths) - 20} additional paths not displayed.")
    def render_comparison() -> None:
        schedule_map: dict[str, sch.ProjectSchedule] = {
            "Baseline": baseline,
            "Current forecast": current,
        }
        for item in st.session_state.scenarios:
            schedule_map[item.name] = item.schedule
        st.dataframe(
            scenario_manager.compare_schedules(schedule_map),
            width="stretch",
            alt="Baseline, current forecast and saved scenario completion comparison",
            hide_index=True,
        )
        for item in st.session_state.scenarios:
            with st.expander(f"{item.name} impact details"):
                impact = item.impact
                cols = st.columns(3)
                cols[0].metric(
                    "Completion slip",
                    f"{impact['project_finish_slip_days']} calendar days",
                )
                cols[1].metric(
                    "Affected activities", impact["affected_activity_count"]
                )
                cols[2].metric(
                    "Activities with float consumed",
                    len(impact["float_consumed_activities"]),
                )
                st.write(
                    "**Events:** "
                    + ", ".join(
                        f"{activity_id} +{days}d"
                        for activity_id, days in item.delays.items()
                    )
                )
                st.write(
                    "**Revised critical path:** "
                    + " → ".join(item.schedule.critical_path)
                )
                if impact["affected_activities"]:
                    st.dataframe(
                        pd.DataFrame(impact["affected_activities"]),
                        width="stretch",
                        alt=f"Activity dates and float changes in {item.name}",
                        hide_index=True,
                    )
        if scenario:
            recommendations = scenario_manager.rule_based_recovery_review(
                scenario["schedule"]
            )
            st.subheader("Rule-based recovery review")
            st.caption(
                "These are review priorities, not optimized recovery actions. "
                "No resource-capacity model is implemented."
            )
            st.dataframe(
                pd.DataFrame(recommendations),
                width="stretch",
                alt="CPM activities recommended for human review",
                hide_index=True,
            )
        if st.session_state.scenarios:
            selected_name = st.selectbox(
                "Remove saved scenario",
                [item.name for item in st.session_state.scenarios],
            )
            if st.button("Remove scenario"):
                st.session_state.scenarios = [
                    item for item in st.session_state.scenarios
                    if item.name != selected_name
                ]
                st.rerun()

    renderers = [
        lambda: _render_overview(frame, baseline, current),
        lambda: _render_register(frame, current),
        render_gantt,
        render_critical_path,
        lambda: _render_delay_analysis(frame, current),
        lambda: _render_progress(frame, current),
        lambda: _render_sequence_validation(frame),
        render_comparison,
        lambda: _render_import_export(frame),
        lambda: _render_assistant(frame),
    ]
    for tab, render in zip(tabs, renderers):
        if tab.open:
            with tab:
                render()


if __name__ == "__main__":
    main()
