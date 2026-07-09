from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_wait_times(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["case_id", "timestamp", "event_id"])
    previous_end = frame.groupby("case_id")["timestamp"].shift() + pd.to_timedelta(
        frame.groupby("case_id")["duration_minutes"].shift(), unit="m"
    )
    frame["wait_hours"] = (
        (frame["timestamp"] - previous_end).dt.total_seconds().div(3600).clip(lower=0)
    ).fillna(0)
    return frame


def calculate_activity_durations(events: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    frame = calculate_wait_times(events)
    case_fields = cases[["case_id", "sla_breached"]]
    frame = frame.merge(case_fields, on="case_id", how="left")
    occurrence = frame.groupby(["case_id", "activity"])["event_id"].transform("count")
    frame["is_rework_event"] = occurrence > 1
    total_wait = max(float(frame["wait_hours"].sum()), 1e-9)
    breached_total = max(int(cases["sla_breached"].sum()), 1)
    result = frame.groupby("activity", as_index=False).agg(
        total_cases=("case_id", "nunique"),
        event_count=("event_id", "count"),
        avg_duration_minutes=("duration_minutes", "mean"),
        median_duration_minutes=("duration_minutes", "median"),
        p90_duration_minutes=("duration_minutes", lambda value: value.quantile(0.90)),
        p95_duration_minutes=("duration_minutes", lambda value: value.quantile(0.95)),
        avg_wait_hours=("wait_hours", "mean"),
        total_wait_hours=("wait_hours", "sum"),
        breached_cases=("sla_breached", "sum"),
        rework_events=("is_rework_event", "sum"),
        labor_cost=(
            "duration_minutes",
            lambda value: float(value.sum() / 60 * 35),
        ),
    )
    result["delay_contribution"] = result["total_wait_hours"] / total_wait
    result["sla_breach_contribution"] = (
        result["breached_cases"] / result["event_count"].clip(lower=1)
    ) * (result["total_cases"] / breached_total).clip(upper=1)
    result["rework_rate"] = result["rework_events"] / result["event_count"].clip(lower=1)
    return result


def _minmax(series: pd.Series) -> pd.Series:
    minimum, maximum = float(series.min()), float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def calculate_bottleneck_score(activity_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = activity_metrics.copy()
    frame["bottleneck_score"] = 100 * (
        0.30 * _minmax(frame["avg_duration_minutes"])
        + 0.25 * _minmax(frame["p95_duration_minutes"])
        + 0.20 * _minmax(frame["avg_wait_hours"])
        + 0.15 * _minmax(frame["sla_breach_contribution"])
        + 0.10 * _minmax(frame["rework_rate"])
    )
    return frame


def rank_bottlenecks(events: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    return calculate_bottleneck_score(calculate_activity_durations(events, cases)).sort_values(
        "bottleneck_score", ascending=False, ignore_index=True
    )
