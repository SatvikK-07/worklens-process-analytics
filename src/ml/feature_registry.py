from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.ml.early_case_features import CATEGORICAL_FEATURES, EARLY_NUMERIC_FEATURES
from src.ml.historical_provider_features import HISTORICAL_NUMERIC_FEATURES


@dataclass(frozen=True)
class FeatureMetadata:
    feature_name: str
    feature_group: str
    description: str
    available_at_prediction_time: bool
    prediction_stage: str
    source_table: str
    uses_future_information: bool
    safe_for_sla_prediction: bool
    safe_for_remaining_time_prediction: bool
    safe_for_early_anomaly_detection: bool
    safe_for_retrospective_analysis: bool
    notes: str = ""


def _safe_feature(name: str) -> FeatureMetadata:
    historical = name in HISTORICAL_NUMERIC_FEATURES
    return FeatureMetadata(
        feature_name=name,
        feature_group="historical" if historical else "early_case",
        description=name.replace("_", " ").capitalize(),
        available_at_prediction_time=True,
        prediction_stage="case_start"
        if name
        in {
            "claim_type",
            "priority",
            "region",
            "provider_type",
            "diagnosis_group",
            "procedure_group",
        }
        else "first_5_events",
        source_table="historical closed cases" if historical else "cases/events",
        uses_future_information=False,
        safe_for_sla_prediction=True,
        safe_for_remaining_time_prediction=True,
        safe_for_early_anomaly_detection=True,
        safe_for_retrospective_analysis=True,
        notes=(
            "Computed strictly from cases closed before case creation"
            if historical
            else "Observed no later than the prediction prefix"
        ),
    )


FORBIDDEN_FEATURES = {
    "total_duration_hours": "Final full-case duration",
    "actual_sla_breach": "Observed SLA target",
    "sla_breached": "Observed SLA target",
    "final_status": "Final workflow state",
    "completed_at": "Completion timestamp",
    "closed_at": "Completion timestamp",
    "case_end_time": "Completion timestamp",
    "case_end": "Completion timestamp",
    "total_event_count": "Full-case event count",
    "remaining_time_hours": "Regression target",
    "future_activity": "Activity after prediction prefix",
    "future_transition_count": "Transitions after prediction prefix",
    "total_cost": "Post-completion cost",
    "final_event_count": "Full-case event count",
    "final_unique_teams": "Full-case team count",
    "final_manual_minutes": "Full-case manual effort",
    "final_outcome": "Final workflow outcome",
    "outcome": "Final workflow outcome",
    "anomaly_label": "Retrospective seeded target",
}


def feature_registry() -> list[FeatureMetadata]:
    rows = [
        _safe_feature(name)
        for name in CATEGORICAL_FEATURES + EARLY_NUMERIC_FEATURES + HISTORICAL_NUMERIC_FEATURES
    ]
    rows.extend(
        FeatureMetadata(
            feature_name=name,
            feature_group="post_completion",
            description=description,
            available_at_prediction_time=False,
            prediction_stage="post_completion",
            source_table="cases/events",
            uses_future_information=True,
            safe_for_sla_prediction=False,
            safe_for_remaining_time_prediction=False,
            safe_for_early_anomaly_detection=False,
            safe_for_retrospective_analysis=True,
            notes="Forbidden for all early/prefix models",
        )
        for name, description in FORBIDDEN_FEATURES.items()
    )
    return rows


def feature_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(asdict(row) for row in feature_registry())


def validate_features(feature_names: list[str], task: str) -> None:
    safety_column = {
        "sla_prediction": "safe_for_sla_prediction",
        "remaining_time_prediction": "safe_for_remaining_time_prediction",
        "early_anomaly_detection": "safe_for_early_anomaly_detection",
        "retrospective_analysis": "safe_for_retrospective_analysis",
    }.get(task)
    if safety_column is None:
        raise ValueError(f"Unknown feature-governance task: {task}")
    registry = feature_registry_frame().set_index("feature_name")
    unknown = sorted(set(feature_names) - set(registry.index))
    if unknown:
        raise ValueError(f"Unregistered features for {task}: {unknown}")
    unsafe = sorted(
        feature for feature in feature_names if not bool(registry.loc[feature, safety_column])
    )
    if unsafe:
        raise ValueError(f"Unsafe features for {task}: {unsafe}")
