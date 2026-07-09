from __future__ import annotations

import pandas as pd


def answer_business_question(
    question: str,
    cases: pd.DataFrame,
    automation_candidates: pd.DataFrame,
    providers: pd.DataFrame,
) -> tuple[str, pd.DataFrame | None]:
    """Answer a constrained set of operational questions from governed data."""
    normalized = question.lower().strip()

    if "automat" in normalized or "savings" in normalized:
        top = automation_candidates.iloc[0]
        answer = (
            f"{top['activity']} should be the first automation candidate. It scores "
            f"{top['automation_priority_score']:.0f}/100 and represents an estimated "
            f"${top['estimated_monthly_savings']:,.0f} in monthly savings."
        )
        return answer, automation_candidates.head(5)

    if "prior authorization" in normalized and ("delay" in normalized or "slow" in normalized):
        subset = cases[cases["claim_type"] == "Prior Authorization"]
        answer = (
            f"Prior authorization cases average {subset['total_duration_hours'].mean():.1f} "
            f"hours and breach SLA at {subset['sla_breached'].mean():.1%}. "
            f"{(subset['rework_count'] > 0).mean():.1%} contain rework, making document "
            "completeness and provider clarification the first root causes to address."
        )
        detail = (
            subset.groupby(["priority", "region"], as_index=False)
            .agg(
                cases=("case_id", "count"),
                avg_hours=("total_duration_hours", "mean"),
                breach_rate=("sla_breached", "mean"),
            )
            .sort_values("breach_rate", ascending=False)
        )
        return answer, detail

    if "risk" in normalized or "breach" in normalized:
        risk_column = (
            "sla_breach_probability" if "sla_breach_probability" in cases else "sla_breached"
        )
        high_risk = cases.sort_values(risk_column, ascending=False).head(10).copy()
        high_risk["sla_breach_probability"] = high_risk[risk_column].astype(float)
        if "recommended_action" not in high_risk:
            high_risk["recommended_action"] = "Review in the intervention queue."
        answer = (
            f"The ten highest-risk cases range from "
            f"{high_risk['sla_breach_probability'].min():.1%} to "
            f"{high_risk['sla_breach_probability'].max():.1%} predicted breach risk. "
            "Move these cases into the intervention queue and resolve document issues first."
        )
        return answer, high_risk[
            [
                "case_id",
                "claim_type",
                "priority",
                "sla_breach_probability",
                "recommended_action",
            ]
        ]

    if "provider" in normalized and ("rework" in normalized or "delay" in normalized):
        provider_metrics = (
            cases.groupby("provider_id", as_index=False)
            .agg(
                cases=("case_id", "count"),
                rework_rate=("rework_count", lambda values: (values > 0).mean()),
                breach_rate=("sla_breached", "mean"),
                avg_duration_hours=("total_duration_hours", "mean"),
            )
            .merge(providers[["provider_id", "provider_type"]], on="provider_id")
            .sort_values(["rework_rate", "cases"], ascending=False)
        )
        top = provider_metrics.iloc[0]
        answer = (
            f"{top['provider_id']} has the highest measured rework rate at "
            f"{top['rework_rate']:.1%} across {int(top['cases']):,} cases. "
            "Review its document-error pattern before changing downstream staffing."
        )
        return answer, provider_metrics.head(10)

    answer = (
        "I can answer governed questions about prior-authorization delay, automation "
        "priority, high-risk cases, and provider rework. Choose one of the suggested "
        "questions to keep the analysis tied to validated metrics."
    )
    return answer, None
