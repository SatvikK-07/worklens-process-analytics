from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import settings

AUTOMATION_PROFILE = {
    "Eligibility Check": (0.94, 0.96, 0.92),
    "Member Info Correction": (0.78, 0.74, 0.70),
    "Document Intake": (0.92, 0.88, 0.90),
    "Document Review": (0.75, 0.68, 0.72),
    "Provider Clarification": (0.61, 0.58, 0.62),
    "Medical Necessity Review": (0.46, 0.42, 0.38),
    "Nurse Review": (0.38, 0.35, 0.32),
    "Medical Director Review": (0.22, 0.18, 0.16),
    "Approval": (0.86, 0.90, 0.82),
    "Denial": (0.68, 0.62, 0.55),
    "Payment Processing": (0.95, 0.97, 0.94),
}


def estimate_monthly_savings(
    monthly_volume: float,
    avg_manual_minutes: float,
    hourly_labor_cost: float,
    automation_feasibility: float,
) -> float:
    return monthly_volume * avg_manual_minutes / 60 * hourly_labor_cost * automation_feasibility


def calculate_manual_hours(monthly_volume: float, avg_duration_minutes: float) -> float:
    return monthly_volume * avg_duration_minutes / 60


def _minmax(series: pd.Series) -> pd.Series:
    if np.isclose(series.max(), series.min()):
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def rank_automation_candidates(
    events: pd.DataFrame,
    cases: pd.DataFrame,
    months: int = 6,
    hourly_labor_cost: float = settings.hourly_labor_cost,
) -> pd.DataFrame:
    case_rework = cases.set_index("case_id")["rework_count"]
    frame = events.copy()
    frame["case_rework"] = frame["case_id"].map(case_rework).fillna(0)
    grouped = (
        frame[frame["activity"].isin(AUTOMATION_PROFILE)]
        .groupby("activity", as_index=False)
        .agg(
            frequency=("event_id", "count"),
            avg_duration_minutes=("duration_minutes", "mean"),
            error_rate=("case_rework", lambda values: float((values > 0).mean())),
            system_fragmentation_score=("application_used", lambda values: values.nunique()),
        )
    )
    grouped["monthly_volume"] = grouped["frequency"] / months
    grouped["monthly_manual_hours"] = (
        grouped["monthly_volume"] * grouped["avg_duration_minutes"] / 60
    )
    profile = grouped["activity"].map(AUTOMATION_PROFILE)
    grouped["repetitiveness_score"] = profile.map(lambda values: values[0])
    grouped["rule_based_score"] = profile.map(lambda values: values[1])
    grouped["automation_feasibility"] = profile.map(lambda values: values[2])
    grouped["estimated_monthly_savings"] = grouped.apply(
        lambda row: estimate_monthly_savings(
            row["monthly_volume"],
            row["avg_duration_minutes"],
            hourly_labor_cost,
            row["automation_feasibility"],
        ),
        axis=1,
    )
    grouped["estimated_annual_savings"] = grouped["estimated_monthly_savings"] * 12
    grouped["automation_priority_score"] = 100 * (
        0.25 * _minmax(grouped["frequency"])
        + 0.20 * _minmax(grouped["monthly_manual_hours"])
        + 0.20 * grouped["repetitiveness_score"]
        + 0.15 * grouped["rule_based_score"]
        + 0.10 * _minmax(grouped["error_rate"])
        + 0.10 * _minmax(grouped["estimated_monthly_savings"])
    )
    return grouped.sort_values("automation_priority_score", ascending=False, ignore_index=True)


def estimate_annual_savings(monthly_savings: float) -> float:
    return monthly_savings * 12
