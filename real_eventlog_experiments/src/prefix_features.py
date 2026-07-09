from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from real_eventlog_experiments.src.preprocess_eventlog import (
    case_summary,
    eligible_case_ids,
)

FORBIDDEN_PREFIX_FEATURES = {
    "case_end",
    "final_activity",
    "final_event_count",
    "final_outcome",
    "final_status",
    "future_activity",
    "future_transition_count",
    "long_case",
    "long_case_median",
    "long_case_q75",
    "open_after_7_days",
    "open_after_14_days",
    "open_after_30_days",
    "process_variant",
    "remaining_time_hours",
    "total_case_duration_hours",
    "total_event_count",
}

HISTORICAL_ENCODING_COLUMNS = [
    "last_activity_train_median_duration",
    "last_activity_train_long_case_rate",
    "prefix_path_train_long_case_rate_smoothed",
    "activity_set_train_median_duration",
    "resource_train_median_duration",
    "resource_train_long_case_rate",
]


@dataclass(frozen=True)
class HistoricalPrefixEncoders:
    target_column: str
    smoothing: float
    global_duration_median: float
    global_target_rate: float
    rate_stats: dict[str, dict[str, tuple[float, int]]]
    duration_values: dict[str, dict[str, tuple[float, ...]]]
    rare_activities: frozenset[str]
    high_delay_activities: frozenset[str]


def prefix_feature_columns(prefix_length: int) -> tuple[list[str], list[str]]:
    categorical = [
        *[f"activity_{index}" for index in range(1, prefix_length + 1)],
        "first_activity",
        "last_activity",
        "last_resource",
        "prefix_path",
        "activity_set",
    ]
    numeric = [
        "prefix_len",
        "elapsed_hours",
        "unique_activities_so_far",
        "observed_rework_count_so_far",
        "observed_activity_repeat_count",
        "observed_transition_count",
        "unique_transitions_so_far",
        "unique_resources_so_far",
        *[f"gap_{index}_{index + 1}_hours" for index in range(1, prefix_length)],
        "mean_gap_hours",
        "max_gap_hours",
        "min_gap_hours",
        "std_gap_hours",
        "start_hour",
        "start_dayofweek",
        "start_month",
        "weekend_start",
        "observed_rare_activity_flag",
        "observed_high_delay_activity_flag",
        *HISTORICAL_ENCODING_COLUMNS,
    ]
    return categorical, numeric


def _ordered_events(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.copy()
    if "resource" not in ordered:
        ordered["resource"] = "Unknown"
    ordered["resource"] = ordered["resource"].fillna("Unknown").astype(str)
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    sort_columns = ["case_id", "timestamp"]
    if "event_id" in ordered:
        sort_columns.append("event_id")
    return ordered.sort_values(sort_columns, kind="mergesort")


def build_prefix_features(
    events: pd.DataFrame,
    prefix_length: int = 3,
) -> pd.DataFrame:
    if prefix_length < 1:
        raise ValueError("prefix_length must be at least 1")
    ordered_all = _ordered_events(events)
    eligible = eligible_case_ids(ordered_all, prefix_length)
    ordered = ordered_all[ordered_all["case_id"].isin(eligible)]
    summary = case_summary(ordered)
    rows: list[dict[str, Any]] = []
    for case_id, group in ordered.groupby("case_id", sort=False):
        prefix = group.head(prefix_length)
        activities = prefix["activity"].astype(str).tolist()
        timestamps = pd.to_datetime(prefix["timestamp"], utc=True)
        gaps = timestamps.diff().dt.total_seconds().div(3600).dropna()
        resources = prefix["resource"].astype(str)
        transitions = list(zip(activities[:-1], activities[1:], strict=True))
        counts = pd.Series(activities).value_counts().sort_index()
        row: dict[str, Any] = {
            "case_id": case_id,
            "case_start": timestamps.iloc[0],
            "case_start_time": timestamps.iloc[0],
            "prefix_timestamp": timestamps.iloc[-1],
            "prefix_len": len(prefix),
            "prefix_length": len(prefix),
            "elapsed_hours": (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600,
            "elapsed_time_so_far": (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
            / 3600,
            "first_activity": activities[0],
            "last_activity": activities[-1],
            "last_resource": resources.iloc[-1],
            "prefix_path": " → ".join(activities),
            "activity_set": " | ".join(sorted(set(activities))),
            "observed_activity_counts": " | ".join(
                f"{activity}:{count}" for activity, count in counts.items()
            ),
            "unique_activities_so_far": len(set(activities)),
            "observed_rework_count_so_far": len(activities) - len(set(activities)),
            "observed_activity_repeat_count": len(activities) - len(set(activities)),
            "repeated_activity_count_so_far": len(activities) - len(set(activities)),
            "observed_transition_count": len(transitions),
            "unique_transitions_so_far": len(set(transitions)),
            "unique_resources_so_far": resources.nunique(),
            "mean_gap_hours": float(gaps.mean()) if not gaps.empty else 0.0,
            "max_gap_hours": float(gaps.max()) if not gaps.empty else 0.0,
            "min_gap_hours": float(gaps.min()) if not gaps.empty else 0.0,
            "std_gap_hours": float(gaps.std(ddof=0)) if not gaps.empty else 0.0,
            "avg_delta_hours": float(gaps.mean()) if not gaps.empty else 0.0,
            "max_delta_hours": float(gaps.max()) if not gaps.empty else 0.0,
            "start_hour": timestamps.iloc[0].hour,
            "start_dayofweek": timestamps.iloc[0].dayofweek,
            "start_day_of_week": timestamps.iloc[0].dayofweek,
            "start_month": timestamps.iloc[0].month,
            "weekend_start": int(timestamps.iloc[0].dayofweek >= 5),
        }
        row.update(
            {f"activity_{index}": activity for index, activity in enumerate(activities, start=1)}
        )
        row.update(
            {
                f"gap_{index}_{index + 1}_hours": float(value)
                for index, value in enumerate(gaps, start=1)
            }
        )
        rows.append(row)
    features = pd.DataFrame(rows).merge(
        summary.drop(columns="case_start"),
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    features["remaining_time_hours"] = (
        features["case_end"] - features["prefix_timestamp"]
    ).dt.total_seconds() / 3600
    return features


def _with_targets(
    train_cases: pd.DataFrame,
    train_prefixes: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    required = {"total_case_duration_hours", target_column}
    if required.issubset(train_prefixes.columns):
        return train_prefixes.copy()
    targets = train_cases[["case_id", *sorted(required)]].drop_duplicates("case_id")
    return train_prefixes.merge(targets, on="case_id", how="left", validate="one_to_one")


def _activity_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (
            column
            for column in frame
            if column.startswith("activity_") and column.removeprefix("activity_").isdigit()
        ),
        key=lambda value: int(value.split("_")[1]),
    )


def _rate_stats(frame: pd.DataFrame, key: str, target: str) -> dict[str, tuple[float, int]]:
    grouped = frame.groupby(key, dropna=False)[target].agg(["sum", "count"])
    return {str(index): (float(row["sum"]), int(row["count"])) for index, row in grouped.iterrows()}


def _duration_values(frame: pd.DataFrame, key: str) -> dict[str, tuple[float, ...]]:
    return {
        str(index): tuple(float(value) for value in values)
        for index, values in frame.groupby(key, dropna=False)["total_case_duration_hours"]
    }


def fit_historical_prefix_encoders(
    train_cases: pd.DataFrame,
    train_prefixes: pd.DataFrame,
    target_column: str = "long_case",
    smoothing: float = 20.0,
) -> HistoricalPrefixEncoders:
    frame = _with_targets(train_cases, train_prefixes, target_column)
    if frame.empty:
        raise ValueError("Cannot fit historical prefix encoders on an empty frame")
    activity_columns = _activity_columns(frame)
    activity_values = frame[activity_columns].stack().astype(str)
    minimum_count = max(2, ceil(len(frame) * 0.01))
    rare_activities = frozenset(
        activity_values.value_counts().loc[lambda counts: counts < minimum_count].index
    )

    activity_gap_rows = []
    for position in range(2, len(activity_columns) + 1):
        gap_column = f"gap_{position - 1}_{position}_hours"
        if gap_column not in frame:
            continue
        activity_gap_rows.append(
            frame[[f"activity_{position}", gap_column]].rename(
                columns={
                    f"activity_{position}": "activity",
                    gap_column: "gap_hours",
                }
            )
        )
    if activity_gap_rows:
        activity_gaps = pd.concat(activity_gap_rows, ignore_index=True)
        global_high_gap = float(activity_gaps["gap_hours"].quantile(0.75))
        medians = activity_gaps.groupby("activity")["gap_hours"].median()
        high_delay_activities = frozenset(medians[medians > global_high_gap].index.astype(str))
    else:
        high_delay_activities = frozenset()

    rate_keys = {
        "last_activity": "last_activity",
        "prefix_path": "prefix_path",
        "last_resource": "last_resource",
    }
    duration_keys = {
        "last_activity": "last_activity",
        "activity_set": "activity_set",
        "last_resource": "last_resource",
    }
    return HistoricalPrefixEncoders(
        target_column=target_column,
        smoothing=float(smoothing),
        global_duration_median=float(frame["total_case_duration_hours"].median()),
        global_target_rate=float(frame[target_column].mean()),
        rate_stats={
            name: _rate_stats(frame, column, target_column) for name, column in rate_keys.items()
        },
        duration_values={
            name: _duration_values(frame, column) for name, column in duration_keys.items()
        },
        rare_activities=rare_activities,
        high_delay_activities=high_delay_activities,
    )


def _smoothed_rate(
    stats: tuple[float, int] | None,
    global_rate: float,
    smoothing: float,
    row_target: float | None,
) -> float:
    if stats is None:
        return global_rate
    total, count = stats
    if row_target is not None:
        total -= row_target
        count -= 1
    if count <= 0:
        return global_rate
    return (total + smoothing * global_rate) / (count + smoothing)


def _leave_one_out_median(
    values: tuple[float, ...] | None,
    default: float,
    row_duration: float | None,
) -> float:
    if not values:
        return default
    candidates = list(values)
    if row_duration is not None and len(candidates) > 1:
        nearest = int(np.argmin(np.abs(np.asarray(candidates) - row_duration)))
        candidates.pop(nearest)
    elif row_duration is not None:
        return default
    return float(np.median(candidates))


def transform_with_historical_prefix_encoders(
    prefixes: pd.DataFrame,
    encoders: HistoricalPrefixEncoders,
    *,
    leave_one_out: bool = False,
) -> pd.DataFrame:
    transformed = prefixes.copy()
    activity_columns = _activity_columns(transformed)
    transformed["observed_rare_activity_flag"] = transformed[activity_columns].apply(
        lambda row: int(any(str(value) in encoders.rare_activities for value in row)),
        axis=1,
    )
    transformed["observed_high_delay_activity_flag"] = transformed[activity_columns].apply(
        lambda row: int(any(str(value) in encoders.high_delay_activities for value in row)),
        axis=1,
    )

    duration_column = (
        transformed["total_case_duration_hours"]
        if leave_one_out and "total_case_duration_hours" in transformed
        else pd.Series([None] * len(transformed), index=transformed.index)
    )
    target_values = (
        transformed[encoders.target_column]
        if leave_one_out and encoders.target_column in transformed
        else pd.Series([None] * len(transformed), index=transformed.index)
    )
    for index, row in transformed.iterrows():
        row_duration = (
            float(duration_column.loc[index]) if pd.notna(duration_column.loc[index]) else None
        )
        row_target = float(target_values.loc[index]) if pd.notna(target_values.loc[index]) else None
        last_activity = str(row["last_activity"])
        prefix_path = str(row["prefix_path"])
        activity_set = str(row["activity_set"])
        resource = str(row["last_resource"])
        transformed.loc[index, "last_activity_train_median_duration"] = _leave_one_out_median(
            encoders.duration_values["last_activity"].get(last_activity),
            encoders.global_duration_median,
            row_duration,
        )
        transformed.loc[index, "activity_set_train_median_duration"] = _leave_one_out_median(
            encoders.duration_values["activity_set"].get(activity_set),
            encoders.global_duration_median,
            row_duration,
        )
        transformed.loc[index, "resource_train_median_duration"] = _leave_one_out_median(
            encoders.duration_values["last_resource"].get(resource),
            encoders.global_duration_median,
            row_duration,
        )
        transformed.loc[index, "last_activity_train_long_case_rate"] = _smoothed_rate(
            encoders.rate_stats["last_activity"].get(last_activity),
            encoders.global_target_rate,
            encoders.smoothing,
            row_target,
        )
        transformed.loc[index, "prefix_path_train_long_case_rate_smoothed"] = _smoothed_rate(
            encoders.rate_stats["prefix_path"].get(prefix_path),
            encoders.global_target_rate,
            encoders.smoothing,
            row_target,
        )
        transformed.loc[index, "resource_train_long_case_rate"] = _smoothed_rate(
            encoders.rate_stats["last_resource"].get(resource),
            encoders.global_target_rate,
            encoders.smoothing,
            row_target,
        )
    return transformed


def assert_prefix_safe(feature_columns: list[str]) -> None:
    unsafe = FORBIDDEN_PREFIX_FEATURES.intersection(feature_columns)
    if unsafe:
        raise ValueError(f"Prefix leakage detected: {sorted(unsafe)}")
