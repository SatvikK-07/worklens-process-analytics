from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import SLA_THRESHOLDS, settings
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

CLAIM_TYPES = tuple(SLA_THRESHOLDS)
REGIONS = ("Northeast", "Southeast", "Midwest", "Southwest", "West")
TEAMS = (
    "Intake Operations",
    "Eligibility Operations",
    "Clinical Review",
    "Medical Directors",
    "Provider Services",
    "Claims Adjudication",
    "Payment Operations",
)
APPLICATIONS = {
    "Claim Received": "ClaimsCore",
    "Eligibility Check": "Member360",
    "Member Info Correction": "Member360",
    "Document Intake": "DocuScan",
    "Document Review": "DocuScan",
    "Provider Clarification": "ProviderPortal",
    "Medical Necessity Review": "PolicyRules",
    "Nurse Review": "PolicyRules",
    "Medical Director Review": "PolicyRules",
    "Approval": "ClaimsCore",
    "Denial": "ClaimsCore",
    "Payment Processing": "PaymentHub",
    "Case Closed": "ClaimsCore",
}
ACTIVITY_TEAMS = {
    "Claim Received": "Intake Operations",
    "Eligibility Check": "Eligibility Operations",
    "Member Info Correction": "Eligibility Operations",
    "Document Intake": "Intake Operations",
    "Document Review": "Claims Adjudication",
    "Provider Clarification": "Provider Services",
    "Medical Necessity Review": "Clinical Review",
    "Nurse Review": "Clinical Review",
    "Medical Director Review": "Medical Directors",
    "Approval": "Claims Adjudication",
    "Denial": "Claims Adjudication",
    "Payment Processing": "Payment Operations",
    "Case Closed": "Claims Adjudication",
}
BASE_MINUTES = {
    "Claim Received": 4,
    "Eligibility Check": 12,
    "Member Info Correction": 22,
    "Document Intake": 9,
    "Document Review": 42,
    "Provider Clarification": 28,
    "Medical Necessity Review": 68,
    "Nurse Review": 48,
    "Medical Director Review": 95,
    "Approval": 13,
    "Denial": 18,
    "Payment Processing": 16,
    "Case Closed": 5,
}
BASE_WAIT_HOURS = {
    "Claim Received": 0.05,
    "Eligibility Check": 0.4,
    "Member Info Correction": 3.0,
    "Document Intake": 0.7,
    "Document Review": 3.5,
    "Provider Clarification": 10.0,
    "Medical Necessity Review": 7.5,
    "Nurse Review": 6.0,
    "Medical Director Review": 18.0,
    "Approval": 1.2,
    "Denial": 1.8,
    "Payment Processing": 4.0,
    "Case Closed": 0.3,
}
TEAM_SPEED = {
    "Intake Operations": 0.92,
    "Eligibility Operations": 0.84,
    "Clinical Review": 1.16,
    "Medical Directors": 1.48,
    "Provider Services": 1.28,
    "Claims Adjudication": 1.04,
    "Payment Operations": 0.96,
}
ROLE_BY_TEAM = {
    "Intake Operations": "Intake Specialist",
    "Eligibility Operations": "Eligibility Analyst",
    "Clinical Review": "Registered Nurse Reviewer",
    "Medical Directors": "Medical Director",
    "Provider Services": "Provider Liaison",
    "Claims Adjudication": "Claims Examiner",
    "Payment Operations": "Payment Specialist",
}


@dataclass(frozen=True)
class GenerationConfig:
    case_count: int = 50_000
    user_count: int = 100
    provider_count: int = 25
    seed: int = 42
    start_date: str = "2025-07-01"
    end_date: str = "2025-12-31"
    output_dir: Path = settings.data_dir


def calculate_sla_breach(total_duration: float, threshold: float) -> int:
    return int(total_duration > threshold)


def generate_users(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    team_weights = np.array([0.12, 0.14, 0.20, 0.08, 0.12, 0.23, 0.11])
    teams = rng.choice(TEAMS, config.user_count, p=team_weights)
    return pd.DataFrame(
        {
            "user_id": [f"U-{index:04d}" for index in range(1, config.user_count + 1)],
            "team": teams,
            "role": [ROLE_BY_TEAM[team] for team in teams],
            "region": rng.choice(REGIONS, config.user_count),
            "experience_level": rng.choice(
                ("Associate", "Intermediate", "Senior"),
                config.user_count,
                p=(0.28, 0.48, 0.24),
            ),
        }
    )


def generate_providers(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider_id": [f"P-{index:04d}" for index in range(1, config.provider_count + 1)],
            "provider_type": rng.choice(
                ("Hospital", "Clinic", "Specialist", "Pharmacy", "Independent Practice"),
                config.provider_count,
                p=(0.20, 0.28, 0.24, 0.12, 0.16),
            ),
            "region": rng.choice(REGIONS, config.provider_count),
            "historical_delay_rate": np.round(
                np.clip(rng.beta(2.2, 6.5, config.provider_count), 0.03, 0.75), 4
            ),
            "document_error_rate": np.round(
                np.clip(rng.beta(1.8, 8.0, config.provider_count), 0.02, 0.62), 4
            ),
        }
    )


def _base_path(claim_type: str, rng: np.random.Generator) -> list[str]:
    path = ["Claim Received", "Eligibility Check", "Document Intake", "Document Review"]
    if claim_type in {
        "Prior Authorization",
        "Appeal",
        "High-Cost Specialty Drug",
        "Out-of-Network Claim",
    }:
        path.append("Medical Necessity Review")
    if claim_type in {"Prior Authorization", "Appeal", "High-Cost Specialty Drug"}:
        path.append("Nurse Review")
    if claim_type in {"Appeal", "High-Cost Specialty Drug"} or (
        claim_type == "Prior Authorization" and rng.random() < 0.30
    ):
        path.append("Medical Director Review")
    decision = (
        "Denial"
        if rng.random()
        < {
            "Medical Claim": 0.10,
            "Pharmacy Claim": 0.08,
            "Prior Authorization": 0.18,
            "Appeal": 0.34,
            "High-Cost Specialty Drug": 0.22,
            "Out-of-Network Claim": 0.20,
        }[claim_type]
        else "Approval"
    )
    path.append(decision)
    if decision == "Approval":
        path.append("Payment Processing")
    path.append("Case Closed")
    return path


def _add_rework(
    path: list[str],
    document_error_rate: float,
    rng: np.random.Generator,
    force_heavy: bool = False,
) -> tuple[list[str], int]:
    rework_count = 0
    result: list[str] = []
    for activity in path:
        result.append(activity)
        if activity == "Eligibility Check" and rng.random() < 0.035:
            result.extend(["Member Info Correction", "Eligibility Check"])
            rework_count += 1
        elif activity == "Document Review" and (
            rng.random() < 0.075 + document_error_rate * 0.26 or force_heavy
        ):
            repeats = 4 if force_heavy else 1
            for _ in range(repeats):
                result.extend(["Provider Clarification", "Document Review"])
                rework_count += 1
        elif activity == "Medical Necessity Review" and rng.random() < 0.055:
            result.extend(["Nurse Review", "Medical Necessity Review"])
            rework_count += 1
    return result, rework_count


def generate_dataset(config: GenerationConfig) -> dict[str, pd.DataFrame]:
    """Create realistic case and event tables using a deterministic random seed."""
    rng = np.random.default_rng(config.seed)
    users = generate_users(config, rng)
    providers = generate_providers(config, rng)
    users_by_team = {team: users.loc[users["team"] == team, "user_id"].to_numpy() for team in TEAMS}
    provider_lookup = providers.set_index("provider_id")

    claim_type_values = rng.choice(
        CLAIM_TYPES,
        config.case_count,
        p=(0.32, 0.18, 0.24, 0.08, 0.08, 0.10),
    )
    provider_values = rng.choice(providers["provider_id"], config.case_count)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    created_offsets = rng.integers(0, int((end - start).total_seconds()), config.case_count)
    created_values = start + pd.to_timedelta(created_offsets, unit="s")
    anomaly_indices = set(
        rng.choice(config.case_count, max(1, int(config.case_count * 0.012)), replace=False)
    )
    open_indices = set(
        rng.choice(config.case_count, max(1, int(config.case_count * 0.045)), replace=False)
    )

    cases: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    event_number = 1

    for index in range(config.case_count):
        case_id = f"C-{index + 1:07d}"
        claim_type = str(claim_type_values[index])
        priority = str(rng.choice(("Standard", "High", "Urgent"), p=(0.72, 0.21, 0.07)))
        region = str(rng.choice(REGIONS))
        provider_id = str(provider_values[index])
        provider = provider_lookup.loc[provider_id]
        is_anomaly = index in anomaly_indices
        is_open = index in open_indices
        path = _base_path(claim_type, rng)
        path, rework_count = _add_rework(
            path,
            float(provider["document_error_rate"]),
            rng,
            force_heavy=is_anomaly and index % 4 == 0,
        )
        if is_open:
            cutoff = int(rng.integers(3, max(4, len(path))))
            path = path[:cutoff]

        current_time = pd.Timestamp(created_values[index])
        active_minutes = 0.0
        case_event_start = len(events)
        priority_factor = {"Standard": 1.0, "High": 0.73, "Urgent": 0.52}[priority]
        provider_factor = 1 + float(provider["historical_delay_rate"]) * 1.25

        for step_index, activity in enumerate(path):
            team = ACTIVITY_TEAMS[activity]
            duration = max(
                1.0,
                rng.lognormal(
                    np.log(BASE_MINUTES[activity]),
                    0.38 if activity != "Medical Director Review" else 0.52,
                )
                * TEAM_SPEED[team],
            )
            wait = rng.gamma(
                shape=1.8,
                scale=(BASE_WAIT_HOURS[activity] / 1.8) * priority_factor * provider_factor,
            )
            if is_anomaly and index % 4 == 1 and step_index == 2:
                wait += 125
            if is_anomaly and index % 4 == 2 and activity == "Medical Director Review":
                wait += 170
            current_time += pd.Timedelta(hours=float(wait))
            user_pool = users_by_team[team]
            user_id = str(rng.choice(user_pool))
            events.append(
                {
                    "event_id": f"E-{event_number:09d}",
                    "case_id": case_id,
                    "activity": activity,
                    "timestamp": current_time,
                    "duration_minutes": round(float(duration), 2),
                    "user_id": user_id,
                    "team": team,
                    "application_used": APPLICATIONS[activity],
                    "screen_name": activity.replace(" ", "_").lower(),
                    "event_type": (
                        "Case Created"
                        if activity == "Claim Received"
                        else "Case Completed"
                        if activity == "Case Closed"
                        else "Task Completed"
                    ),
                    "status": (
                        "Closed"
                        if activity == "Case Closed"
                        else "Pending"
                        if step_index == len(path) - 1 and is_open
                        else "Completed"
                    ),
                }
            )
            event_number += 1
            active_minutes += float(duration)
            current_time += pd.Timedelta(minutes=float(duration))

        if is_anomaly and index % 4 == 3 and len(events) - case_event_start >= 3:
            segment = events[case_event_start:]
            approval_positions = [
                offset for offset, row in enumerate(segment) if row["activity"] == "Approval"
            ]
            payment_positions = [
                offset
                for offset, row in enumerate(segment)
                if row["activity"] == "Payment Processing"
            ]
            if approval_positions and payment_positions:
                a_pos, p_pos = approval_positions[0], payment_positions[0]
                segment[a_pos]["timestamp"], segment[p_pos]["timestamp"] = (
                    segment[p_pos]["timestamp"],
                    segment[a_pos]["timestamp"],
                )

        total_hours = max(0.1, (current_time - created_values[index]).total_seconds() / 3600)
        threshold = 24 if priority == "Urgent" else SLA_THRESHOLDS[claim_type]
        breached = calculate_sla_breach(total_hours, threshold)
        outcome = (
            "Pending Info"
            if is_open
            else "Denied"
            if "Denial" in path
            else "Paid"
            if "Payment Processing" in path
            else "Approved"
        )
        breach_hours = max(0.0, total_hours - threshold)
        total_cost = active_minutes / 60 * settings.hourly_labor_cost + breach_hours * 12
        cases.append(
            {
                "case_id": case_id,
                "claim_type": claim_type,
                "priority": priority,
                "region": region,
                "provider_id": provider_id,
                "member_id": f"M-{int(rng.integers(1, config.case_count * 2)):08d}",
                "diagnosis_group": str(
                    rng.choice(
                        (
                            "Cardiology",
                            "Oncology",
                            "Orthopedics",
                            "Behavioral Health",
                            "Endocrinology",
                            "Respiratory",
                            "General",
                        )
                    )
                ),
                "procedure_group": str(
                    rng.choice(
                        (
                            "Diagnostic",
                            "Surgical",
                            "Therapy",
                            "Medication",
                            "Imaging",
                            "Inpatient",
                            "Outpatient",
                        )
                    )
                ),
                "created_at": created_values[index],
                "closed_at": pd.NaT if is_open else current_time,
                "outcome": outcome,
                "sla_threshold_hours": threshold,
                "total_duration_hours": round(total_hours, 2),
                "sla_breached": breached,
                "total_cost": round(total_cost, 2),
                "rework_count": rework_count,
                "anomaly_label": int(is_anomaly),
            }
        )

    return {
        "cases": pd.DataFrame(cases),
        "events": pd.DataFrame(events),
        "users": users,
        "providers": providers.reset_index(drop=True),
    }


def validate_dataset(tables: dict[str, pd.DataFrame], minimum_cases: int) -> dict[str, object]:
    cases, events = tables["cases"], tables["events"]
    chronological = (
        events.sort_values(["case_id", "timestamp"])
        .groupby("case_id")["timestamp"]
        .apply(lambda values: values.is_monotonic_increasing)
        .mean()
    )
    report = {
        "case_count": int(len(cases)),
        "event_count": int(len(events)),
        "user_count": int(len(tables["users"])),
        "provider_count": int(len(tables["providers"])),
        "sla_breach_rate": round(float(cases["sla_breached"].mean()), 4),
        "rework_case_rate": round(float((cases["rework_count"] > 0).mean()), 4),
        "anomaly_case_rate": round(float(cases["anomaly_label"].mean()), 4),
        "timestamp_span_days": int((events["timestamp"].max() - events["timestamp"].min()).days),
        "chronological_case_rate": round(float(chronological), 4),
    }
    failures = []
    if len(cases) < minimum_cases:
        failures.append(f"Expected at least {minimum_cases:,} cases")
    if minimum_cases >= 50_000 and len(events) < 300_000:
        failures.append("Expected at least 300,000 events")
    if cases["sla_breached"].mean() < 0.10:
        failures.append("SLA breach rate is below 10%")
    if (cases["rework_count"] > 0).mean() < 0.05:
        failures.append("Rework case rate is below 5%")
    if cases["anomaly_label"].mean() < 0.01:
        failures.append("Anomaly case rate is below 1%")
    if report["timestamp_span_days"] < 180:
        failures.append("Timestamp coverage is shorter than six months")
    if failures:
        raise ValueError("; ".join(failures))
    return report


def write_dataset(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WorkLens AI demo data.")
    parser.add_argument("--cases", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--output-dir", type=Path, default=settings.data_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        case_count=args.cases,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    LOGGER.info("Generating %s cases", f"{config.case_count:,}")
    tables = generate_dataset(config)
    report = validate_dataset(tables, minimum_cases=config.case_count)
    paths = write_dataset(tables, config.output_dir)
    (config.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("Synthetic data written to %s", config.output_dir)
    LOGGER.info("Validation report: %s", report)
    for name, path in paths.items():
        LOGGER.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
