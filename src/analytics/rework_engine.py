from __future__ import annotations

from collections import defaultdict

import pandas as pd

ROOT_CAUSES = {
    "Provider Clarification": "Incomplete provider submission or missing documents",
    "Member Info Correction": "Manual eligibility or member-data mismatch",
    "Nurse Review": "Clinical policy ambiguity or duplicate review requirement",
}


def detect_rework_loops(events: pd.DataFrame) -> pd.DataFrame:
    """Detect immediate A → B → A loops and aggregate their business impact."""
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["case_id", "timestamp", "event_id"])
    loops: dict[str, list[dict[str, object]]] = defaultdict(list)

    for case_id, group in frame.groupby("case_id", sort=False):
        records = group.to_dict("records")
        for index in range(len(records) - 2):
            first, middle, last = records[index : index + 3]
            if first["activity"] == last["activity"] and first["activity"] != middle["activity"]:
                pattern = f"{first['activity']} → {middle['activity']} → {last['activity']}"
                extra_minutes = float(middle["duration_minutes"]) + float(last["duration_minutes"])
                elapsed_hours = (
                    pd.Timestamp(last["timestamp"]) - pd.Timestamp(first["timestamp"])
                ).total_seconds() / 3600
                loops[pattern].append(
                    {
                        "case_id": case_id,
                        "extra_minutes": extra_minutes,
                        "extra_hours": max(elapsed_hours, extra_minutes / 60),
                        "team": middle["team"],
                        "root_activity": middle["activity"],
                    }
                )

    rows = []
    for pattern, occurrences in loops.items():
        detail = pd.DataFrame(occurrences)
        root_activity = str(detail["root_activity"].mode().iloc[0])
        rows.append(
            {
                "loop_pattern": pattern,
                "occurrences": len(detail),
                "cases_affected": int(detail["case_id"].nunique()),
                "avg_extra_time_hours": float(detail["extra_hours"].mean()),
                "avg_extra_cost": float(detail["extra_minutes"].mean() / 60 * 35),
                "most_common_team": str(detail["team"].mode().iloc[0]),
                "root_cause_hypothesis": ROOT_CAUSES.get(
                    root_activity, "Manual routing or policy-rule ambiguity"
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "loop_pattern",
                "occurrences",
                "cases_affected",
                "avg_extra_time_hours",
                "avg_extra_cost",
                "most_common_team",
                "root_cause_hypothesis",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["cases_affected", "occurrences"], ascending=False, ignore_index=True
    )


def calculate_loop_frequency(events: pd.DataFrame) -> pd.DataFrame:
    return detect_rework_loops(events)[["loop_pattern", "occurrences", "cases_affected"]]


def calculate_extra_time_from_rework(events: pd.DataFrame) -> float:
    loops = detect_rework_loops(events)
    return float((loops["avg_extra_time_hours"] * loops["occurrences"]).sum())


def calculate_rework_cost(events: pd.DataFrame) -> float:
    loops = detect_rework_loops(events)
    return float((loops["avg_extra_cost"] * loops["occurrences"]).sum())


def rank_rework_patterns(events: pd.DataFrame) -> pd.DataFrame:
    return detect_rework_loops(events)
