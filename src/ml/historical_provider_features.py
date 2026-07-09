from __future__ import annotations

import pandas as pd

HISTORICAL_NUMERIC_FEATURES = [
    "historical_provider_delay_rate",
    "historical_provider_rework_rate",
    "provider_history_case_count",
    "historical_team_avg_duration",
    "team_history_case_count",
]


def _provider_history(cases: pd.DataFrame) -> pd.DataFrame:
    completed = cases.dropna(subset=["closed_at"]).copy()
    completed["closed_at"] = pd.to_datetime(completed["closed_at"])
    completed = (
        completed.groupby(["provider_id", "closed_at"], as_index=False)
        .agg(
            cases_at_time=("case_id", "count"),
            breaches_at_time=("sla_breached", "sum"),
            rework_at_time=("rework_count", lambda values: int((values > 0).sum())),
        )
        .sort_values(["provider_id", "closed_at"])
    )
    completed["provider_history_case_count"] = completed.groupby("provider_id")[
        "cases_at_time"
    ].cumsum()
    completed["cumulative_breaches"] = completed.groupby("provider_id")["breaches_at_time"].cumsum()
    completed["cumulative_rework"] = completed.groupby("provider_id")["rework_at_time"].cumsum()
    completed["historical_provider_delay_rate"] = (
        completed["cumulative_breaches"] / completed["provider_history_case_count"]
    )
    completed["historical_provider_rework_rate"] = (
        completed["cumulative_rework"] / completed["provider_history_case_count"]
    )
    return completed[
        [
            "provider_id",
            "closed_at",
            "historical_provider_delay_rate",
            "historical_provider_rework_rate",
            "provider_history_case_count",
        ]
    ].sort_values(["closed_at", "provider_id"])


def _team_history(cases: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    completed_cases = cases.dropna(subset=["closed_at"])[
        ["case_id", "closed_at", "total_duration_hours"]
    ].copy()
    completed_cases["closed_at"] = pd.to_datetime(completed_cases["closed_at"])
    case_teams = events[["case_id", "team"]].drop_duplicates()
    completed = (
        case_teams.merge(completed_cases, on="case_id", how="inner")
        .groupby(["team", "closed_at"], as_index=False)
        .agg(
            cases_at_time=("case_id", "count"),
            duration_at_time=("total_duration_hours", "sum"),
        )
        .sort_values(["team", "closed_at"])
    )
    completed["team_history_case_count"] = completed.groupby("team")["cases_at_time"].cumsum()
    completed["cumulative_duration"] = completed.groupby("team")["duration_at_time"].cumsum()
    completed["historical_team_avg_duration"] = (
        completed["cumulative_duration"] / completed["team_history_case_count"]
    )
    return completed[
        [
            "team",
            "closed_at",
            "historical_team_avg_duration",
            "team_history_case_count",
        ]
    ].sort_values(["closed_at", "team"])


def add_historical_features(
    early_features: pd.DataFrame,
    cases: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Add only records closed strictly before each case was created."""
    features = early_features.copy()
    features["created_at"] = pd.to_datetime(features["created_at"])
    features["_original_order"] = range(len(features))

    provider_history = _provider_history(cases)
    features = pd.merge_asof(
        features.sort_values(["created_at", "provider_id"]),
        provider_history,
        left_on="created_at",
        right_on="closed_at",
        by="provider_id",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns=["closed_at"])

    team_history = _team_history(cases, events)
    features = pd.merge_asof(
        features.sort_values(["created_at", "current_team"]),
        team_history,
        left_on="created_at",
        right_on="closed_at",
        left_by="current_team",
        right_by="team",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns=["closed_at", "team"])

    defaults = {
        "historical_provider_delay_rate": 0.10,
        "historical_provider_rework_rate": 0.15,
        "provider_history_case_count": 0,
        "historical_team_avg_duration": 48.0,
        "team_history_case_count": 0,
    }
    for column, default in defaults.items():
        features[column] = features[column].fillna(default)
    return (
        features.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )
