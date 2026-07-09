from __future__ import annotations

import pandas as pd


def case_summary(events: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["case_id", "timestamp"]
    if "event_id" in events:
        sort_columns.append("event_id")
    ordered = events.sort_values(sort_columns, kind="mergesort")
    summary = ordered.groupby("case_id", as_index=False).agg(
        case_start=("timestamp", "min"),
        case_end=("timestamp", "max"),
        total_event_count=("activity", "size"),
        final_activity=("activity", "last"),
        process_variant=("activity", lambda values: " → ".join(values)),
    )
    summary["total_case_duration_hours"] = (
        summary["case_end"] - summary["case_start"]
    ).dt.total_seconds() / 3600
    return summary


def eligible_case_ids(events: pd.DataFrame, prefix_length: int) -> pd.Index:
    counts = events.groupby("case_id").size()
    return counts[counts >= prefix_length].index
