from __future__ import annotations

import numpy as np
import pandas as pd


def get_total_cases(cases: pd.DataFrame) -> int:
    return int(cases["case_id"].nunique())


def get_completed_cases(cases: pd.DataFrame) -> int:
    return int(cases["closed_at"].notna().sum())


def get_avg_handling_time(cases: pd.DataFrame) -> float:
    return float(cases["total_duration_hours"].mean())


def get_sla_breach_rate(cases: pd.DataFrame) -> float:
    return float(cases["sla_breached"].mean())


def get_rework_rate(cases: pd.DataFrame) -> float:
    return float((cases["rework_count"] > 0).mean())


def get_top_bottleneck(bottlenecks: pd.DataFrame) -> str:
    if bottlenecks.empty:
        return "Not available"
    return str(bottlenecks.sort_values("bottleneck_score", ascending=False).iloc[0]["activity"])


def get_estimated_savings(candidates: pd.DataFrame) -> float:
    if candidates.empty:
        return 0.0
    return float(candidates["estimated_monthly_savings"].sum())


def calculate_kpis(
    cases: pd.DataFrame,
    bottlenecks: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    return {
        "total_cases": get_total_cases(cases),
        "completed_cases": get_completed_cases(cases),
        "open_cases": int(cases["closed_at"].isna().sum()),
        "avg_handling_hours": get_avg_handling_time(cases),
        "median_handling_hours": float(cases["total_duration_hours"].median()),
        "sla_breach_rate": get_sla_breach_rate(cases),
        "rework_rate": get_rework_rate(cases),
        "automation_savings": get_estimated_savings(
            candidates if candidates is not None else pd.DataFrame()
        ),
        "top_bottleneck": get_top_bottleneck(
            bottlenecks if bottlenecks is not None else pd.DataFrame()
        ),
        "high_risk_cases": int(
            cases.get("sla_breach_probability", pd.Series(dtype=float)).ge(0.75).sum()
        ),
        "anomalous_cases": int(cases.get("anomaly_label", pd.Series(dtype=int)).sum()),
    }


def monthly_trends(cases: pd.DataFrame) -> pd.DataFrame:
    frame = cases.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    frame["month"] = frame["created_at"].dt.to_period("M").dt.to_timestamp()
    return (
        frame.groupby("month", as_index=False)
        .agg(
            cases_processed=("case_id", "nunique"),
            avg_handling_hours=("total_duration_hours", "mean"),
            sla_breach_rate=("sla_breached", "mean"),
            rework_rate=("rework_count", lambda values: (values > 0).mean()),
            cost_leakage=(
                "total_duration_hours",
                lambda values: float(np.maximum(values - 48, 0).sum() * 12),
            ),
        )
        .sort_values("month")
    )
