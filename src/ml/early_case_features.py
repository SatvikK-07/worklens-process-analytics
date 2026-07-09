from __future__ import annotations

import pandas as pd

EARLY_EVENT_COUNT = 5

CATEGORICAL_FEATURES = [
    "claim_type",
    "priority",
    "region",
    "provider_type",
    "diagnosis_group",
    "procedure_group",
    "current_activity",
    "current_team",
]

EARLY_NUMERIC_FEATURES = [
    "event_count",
    "handoff_count",
    "unique_teams_count",
    "rework_count",
    "provider_clarification_count",
    "document_review_duration",
    "medical_review_duration",
    "queue_wait_time_total",
    "elapsed_time_so_far",
    "previous_activity_duration",
    "avg_activity_duration",
    "max_activity_duration",
    "missing_document_flag",
]


def build_early_case_features(
    cases: pd.DataFrame,
    events: pd.DataFrame,
    providers: pd.DataFrame,
    first_n_events: int = EARLY_EVENT_COUNT,
) -> pd.DataFrame:
    """Build features that exist after the first N observed events only."""
    event_frame = events.copy()
    event_frame["timestamp"] = pd.to_datetime(event_frame["timestamp"])
    event_frame = event_frame.sort_values(["case_id", "timestamp", "event_id"])
    previous_timestamp = event_frame.groupby("case_id")["timestamp"].shift()
    previous_duration = event_frame.groupby("case_id")["duration_minutes"].shift().fillna(0)
    previous_end = previous_timestamp + pd.to_timedelta(previous_duration, unit="m")
    event_frame["queue_wait_hours"] = (
        (event_frame["timestamp"] - previous_end)
        .dt.total_seconds()
        .div(3600)
        .clip(lower=0)
        .fillna(0)
    )
    event_frame["user_changed"] = (
        event_frame["user_id"] != event_frame.groupby("case_id")["user_id"].shift()
    ).astype(int)

    snapshot = event_frame.groupby("case_id", sort=False).head(first_n_events).copy()
    snapshot["is_rework"] = snapshot.groupby(["case_id", "activity"]).cumcount().gt(0).astype(int)
    aggregate = snapshot.groupby("case_id", as_index=False).agg(
        event_count=("event_id", "count"),
        handoff_count=("user_changed", lambda values: max(int(values.sum()) - 1, 0)),
        unique_teams_count=("team", "nunique"),
        rework_count=("is_rework", "sum"),
        provider_clarification_count=(
            "activity",
            lambda values: int((values == "Provider Clarification").sum()),
        ),
        queue_wait_time_total=("queue_wait_hours", "sum"),
        elapsed_time_so_far=(
            "timestamp",
            lambda values: max(0.0, (values.max() - values.min()).total_seconds() / 3600),
        ),
        previous_activity_duration=("duration_minutes", "last"),
        avg_activity_duration=("duration_minutes", "mean"),
        max_activity_duration=("duration_minutes", "max"),
        current_activity=("activity", "last"),
        current_team=("team", "last"),
    )
    durations = snapshot.pivot_table(
        index="case_id",
        columns="activity",
        values="duration_minutes",
        aggfunc="sum",
        fill_value=0,
    )
    aggregate = aggregate.merge(
        pd.DataFrame(
            {
                "case_id": durations.index,
                "document_review_duration": durations.get(
                    "Document Review", pd.Series(0, index=durations.index)
                ).values,
                "medical_review_duration": (
                    durations.get("Medical Necessity Review", pd.Series(0, index=durations.index))
                    + durations.get("Nurse Review", pd.Series(0, index=durations.index))
                    + durations.get("Medical Director Review", pd.Series(0, index=durations.index))
                ).values,
            }
        ),
        on="case_id",
        how="left",
    )

    provider_fields = providers[["provider_id", "provider_type"]]
    excluded_post_completion = {
        "closed_at",
        "outcome",
        "total_duration_hours",
        "sla_breached",
        "total_cost",
        "rework_count",
        "anomaly_label",
    }
    base_columns = [column for column in cases.columns if column not in excluded_post_completion]
    feature_frame = (
        cases[base_columns]
        .merge(provider_fields, on="provider_id", how="left")
        .merge(aggregate, on="case_id", how="left")
    )
    feature_frame["missing_document_flag"] = (
        feature_frame["provider_clarification_count"].fillna(0) > 0
    ).astype(int)
    for column in EARLY_NUMERIC_FEATURES:
        feature_frame[column] = feature_frame[column].fillna(0)
    return feature_frame
