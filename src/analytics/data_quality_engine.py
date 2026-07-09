from __future__ import annotations

import pandas as pd

from src.utils.config import ALLOWED_ACTIVITIES

ISSUE_WEIGHTS = {
    "Missing case ID": 5.0,
    "Missing timestamp": 5.0,
    "Negative duration": 4.0,
    "Duplicate event": 2.0,
    "Out-of-order event": 3.0,
    "Invalid activity": 3.0,
    "Unknown user": 3.0,
    "Case without closure": 1.0,
    "Payment before approval": 5.0,
    "Case Closed not final": 4.0,
}


def run_data_quality_checks(
    cases: pd.DataFrame,
    events: pd.DataFrame,
    users: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    issues: list[dict[str, object]] = []

    def add_issue(issue: str, mask: pd.Series, recommendation: str) -> None:
        count = int(mask.sum())
        if count:
            sample = frame.loc[mask, "event_id"].head(5).astype(str).tolist()
            issues.append(
                {
                    "issue_type": issue,
                    "affected_records": count,
                    "sample_records": ", ".join(sample),
                    "recommended_fix": recommendation,
                    "severity_weight": ISSUE_WEIGHTS[issue],
                }
            )

    add_issue("Missing case ID", frame["case_id"].isna(), "Reject or recover orphan events.")
    add_issue("Missing timestamp", frame["timestamp"].isna(), "Backfill from source audit logs.")
    add_issue("Negative duration", frame["duration_minutes"] < 0, "Correct duration derivation.")
    add_issue("Duplicate event", frame["event_id"].duplicated(), "Deduplicate by event ID.")
    add_issue(
        "Invalid activity",
        ~frame["activity"].isin(ALLOWED_ACTIVITIES),
        "Map activity to the controlled vocabulary.",
    )
    add_issue(
        "Unknown user",
        ~frame["user_id"].isin(users["user_id"]),
        "Repair the user master-data reference.",
    )

    ordered = frame.sort_values(["case_id", "timestamp", "event_id"])
    last_activity = ordered.groupby("case_id")["activity"].last()
    closed_cases = cases.loc[cases["closed_at"].notna(), "case_id"]
    missing_closure_ids = set(closed_cases) - set(
        last_activity[last_activity == "Case Closed"].index
    )
    if missing_closure_ids:
        issues.append(
            {
                "issue_type": "Case without closure",
                "affected_records": len(missing_closure_ids),
                "sample_records": ", ".join(sorted(missing_closure_ids)[:5]),
                "recommended_fix": "Add or recover the terminal Case Closed event.",
                "severity_weight": ISSUE_WEIGHTS["Case without closure"],
            }
        )

    payment_before = 0
    closed_not_final = 0
    for _, group in frame.groupby("case_id", sort=False):
        activities = group.sort_values("timestamp")["activity"].tolist()
        if "Payment Processing" in activities and "Approval" in activities:
            payment_before += int(
                activities.index("Payment Processing") < activities.index("Approval")
            )
        if "Case Closed" in activities:
            closed_not_final += int(activities[-1] != "Case Closed")
    for issue, count, fix in (
        ("Payment before approval", payment_before, "Investigate invalid adjudication order."),
        ("Case Closed not final", closed_not_final, "Recover reopen events or correct status."),
    ):
        if count:
            issues.append(
                {
                    "issue_type": issue,
                    "affected_records": count,
                    "sample_records": "See affected case export",
                    "recommended_fix": fix,
                    "severity_weight": ISSUE_WEIGHTS[issue],
                }
            )

    issue_frame = pd.DataFrame(issues)
    if issue_frame.empty:
        return 100.0, pd.DataFrame(
            columns=[
                "issue_type",
                "affected_records",
                "sample_records",
                "recommended_fix",
                "severity_weight",
            ]
        )
    penalty = (
        (issue_frame["affected_records"] * issue_frame["severity_weight"]).sum()
        / max(len(events), 1)
        * 100
    )
    return round(max(0.0, 100.0 - float(penalty)), 2), issue_frame
