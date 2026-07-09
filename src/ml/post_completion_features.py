from __future__ import annotations

import pandas as pd

POST_COMPLETION_COLUMNS = [
    "outcome",
    "closed_at",
    "total_duration_hours",
    "sla_breached",
    "total_cost",
    "anomaly_label",
]


def build_post_completion_features(cases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build retrospective features for analytics, never for prediction."""
    event_summary = events.groupby("case_id", as_index=False).agg(
        final_event_count=("event_id", "count"),
        final_unique_teams=("team", "nunique"),
        final_manual_minutes=("duration_minutes", "sum"),
    )
    columns = ["case_id", *[column for column in POST_COMPLETION_COLUMNS if column in cases]]
    return cases[columns].merge(event_summary, on="case_id", how="left")
