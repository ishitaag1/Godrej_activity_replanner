"""Construction-sequence checks for documented, mandatory workflow ordering."""

from __future__ import annotations
from typing import Any

import networkx as nx
import pandas as pd

import scheduler as sch


def _find_ids(frame: pd.DataFrame, token: str) -> list[str]:
    return [
        str(activity_id)
        for activity_id in frame["activity_id"]
        if token.casefold() in str(activity_id).casefold()
    ]


def _same_work_scope(upstream_id: str, downstream_id: str) -> bool:
    upstream = upstream_id.split("-")
    downstream = downstream_id.split("-")
    if upstream[:2] != downstream[:2]:
        return False
    upstream_unit = len(upstream) > 2 and upstream[2].startswith("U")
    downstream_unit = len(downstream) > 2 and downstream[2].startswith("U")
    if upstream_unit or downstream_unit:
        return upstream_unit and downstream_unit and upstream[:3] == downstream[:3]
    return True


def validate_construction_sequence(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Report missing documented dependency paths; never edits user dependencies."""
    graph = sch.build_graph(frame)
    sch.validate_graph(graph)
    rules = [
        (
            "MEP conduit / sleeve installation precedes the related RCC pour",
            "-MEP-CONDUIT",
            "-SLAB-POUR",
            "B.6",
            "High",
        ),
        (
            "RCC post-pour clearance precedes dependent blockwork",
            "-POST-POUR-CLEAR",
            "-BLOCK-BLOCK-WORK",
            "B.6 and C.7",
            "High",
        ),
        (
            "MEP chasing precedes plumbing first fix",
            "-WET-CHASING",
            "-PLUMB-FIRST-FIX",
            "C.8 stage 1",
            "High",
        ),
        (
            "Plumbing first fix precedes wet-area waterproofing",
            "-PLUMB-FIRST-FIX",
            "-WET-WATERPROOF",
            "C.8 stage 1-2",
            "High",
        ),
        (
            "Protective screed precedes wet-area second fix and wall dado finishes",
            "-WET-PROTECTIVE-SCREED",
            "-PLUMB-SECOND-FIX",
            "C.8 stage 2-3",
            "High",
        ),
        (
            "Plumbing second fix precedes wall dado",
            "-PLUMB-SECOND-FIX",
            "-WALL-DADO",
            "C.8 stage 3",
            "High",
        ),
        (
            "Wet-area waterproofing precedes protective screed",
            "-WET-WATERPROOF",
            "-WET-PROTECTIVE-SCREED",
            "C.8 stage 2",
            "High",
        ),
        (
            "Toilet / balcony waterproofing precedes room flooring",
            "-WET-WATERPROOF",
            "-ROOM-FLOORING",
            "C.9 stage 2",
            "High",
        ),
        (
            "Facade frame and sealant completion precedes water testing",
            "-FACADE-FRAME",
            "-FACADE-WATER-TEST",
            "E.12",
            "High",
        ),
        (
            "Facade water testing precedes damp-proof coat",
            "-FACADE-WATER-TEST",
            "-DAMP-PROOF-COAT",
            "E.12",
            "High",
        ),
        (
            "Damp-proof coat precedes external painting",
            "-DAMP-PROOF-COAT",
            "-EXTERNAL-PAINT",
            "E.12",
            "High",
        ),
        (
            "Terrace service clearance precedes terrace waterproofing",
            "-TERRACE-SERVICE-CLEAR",
            "-TERRACE-WATERPROOF",
            "E.13",
            "High",
        ),
        (
            "Terrace RCC completion precedes terrace waterproofing",
            "-TERRACE-RCC-COMPLETE",
            "-TERRACE-WATERPROOF",
            "E.13",
            "High",
        ),
        (
            "Terrace waterproofing protection precedes final inspection",
            "-TERRACE-PROTECTION",
            "-FINAL-INSPECTION",
            "E.13 and E.14",
            "High",
        ),
        (
            "Final inspection follows unit and lobby deep cleaning",
            "-UNIT-DEEP-CLEAN",
            "-FINAL-INSPECTION",
            "E.14",
            "High",
        ),
        (
            "Final inspection follows lobby deep cleaning",
            "-LOBBY-DEEP-CLEAN",
            "-FINAL-INSPECTION",
            "E.14",
            "High",
        ),
    ]
    violations: list[dict[str, Any]] = []
    for rule, upstream_token, downstream_token, reference, severity in rules:
        upstream_ids = _find_ids(frame, upstream_token)
        downstream_ids = _find_ids(frame, downstream_token)
        if not upstream_ids or not downstream_ids:
            continue
        for upstream in upstream_ids:
            for downstream in downstream_ids:
                if upstream == downstream:
                    continue
                # Codes are scoped by tower/floor/unit, so only same-context
                # activities should be compared except final inspection.
                same_scope = _same_work_scope(upstream, downstream) or downstream.endswith(
                    "-FINAL-INSPECTION"
                )
                if not same_scope:
                    continue
                if not nx.has_path(graph, upstream, downstream):
                    upstream_name = frame.loc[
                        frame["activity_id"] == upstream, "activity_name"
                    ].iloc[0]
                    downstream_name = frame.loc[
                        frame["activity_id"] == downstream, "activity_name"
                    ].iloc[0]
                    violations.append(
                        {
                            "activity_id": downstream,
                            "activity_name": downstream_name,
                            "violated_rule": rule,
                            "upstream_activity_id": upstream,
                            "upstream_activity_name": upstream_name,
                            "downstream_activity_id": downstream,
                            "downstream_activity_name": downstream_name,
                            "severity": severity,
                            "recommended_action": (
                                "Review and add an approved finish-to-start dependency "
                                "only if this relationship applies to the project."
                            ),
                            "source_document_reference": (
                                f"Construction_Activity_Sequence.docx §{reference}"
                            ),
                        }
                    )

    # The retained staircase constraint is a project note, not an inferred
    # generic dependency: verify the configured finish is after lobby finishes.
    retained = _find_ids(frame, "-RETAINED-FINISH")
    lobby_finishes = _find_ids(frame, "-LOBBY-WALL")
    for stair_id in retained:
        for lobby_id in lobby_finishes:
            if stair_id.split("-")[0] != lobby_id.split("-")[0]:
                continue
            if not nx.has_path(graph, lobby_id, stair_id):
                violations.append(
                    {
                        "activity_id": stair_id,
                        "activity_name": frame.loc[
                            frame["activity_id"] == stair_id, "activity_name"
                        ].iloc[0],
                        "violated_rule": (
                            "Retain one staircase for material shifting and finish it "
                            "top-to-bottom after common-area finishing"
                        ),
                        "upstream_activity_id": lobby_id,
                        "upstream_activity_name": frame.loc[
                            frame["activity_id"] == lobby_id, "activity_name"
                        ].iloc[0],
                        "downstream_activity_id": stair_id,
                        "downstream_activity_name": frame.loc[
                            frame["activity_id"] == stair_id, "activity_name"
                        ].iloc[0],
                        "severity": "High",
                        "recommended_action": (
                            "Review staircase access and retain one route for material "
                            "movement until common-area finishes are complete."
                        ),
                        "source_document_reference": (
                            "Construction_Activity_Sequence.docx §D.11"
                        ),
                    }
                )
    return violations
