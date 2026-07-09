from __future__ import annotations

import pandas as pd


def _case_paths(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["case_id", "timestamp", "event_id"])
    grouped = frame.groupby("case_id")
    return grouped.agg(
        process_path=("activity", lambda values: " → ".join(values)),
        event_count=("event_id", "count"),
        path_duration_hours=(
            "timestamp",
            lambda values: max(0.0, (values.max() - values.min()).total_seconds() / 3600),
        ),
        manual_minutes=("duration_minutes", "sum"),
    ).reset_index()


def get_top_process_paths(events: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    paths = _case_paths(events)
    return (
        paths.groupby("process_path", as_index=False)
        .agg(
            case_count=("case_id", "count"),
            avg_duration_hours=("path_duration_hours", "mean"),
            avg_manual_minutes=("manual_minutes", "mean"),
        )
        .sort_values("case_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_slowest_paths(events: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        get_top_process_paths(events, n=10_000)
        .sort_values("avg_duration_hours", ascending=False)
        .head(n)
    )


def get_rework_paths(events: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    paths = _case_paths(events)
    paths["has_rework"] = paths["process_path"].map(
        lambda path: len(path.split(" → ")) != len(set(path.split(" → ")))
    )
    return (
        paths[paths["has_rework"]]
        .sort_values(["event_count", "path_duration_hours"], ascending=False)
        .head(n)
    )


def get_rare_paths(events: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    paths = get_top_process_paths(events, n=10_000)
    return paths.sort_values(["case_count", "avg_duration_hours"]).head(n)
