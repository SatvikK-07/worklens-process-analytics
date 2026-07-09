from __future__ import annotations

import pandas as pd

from src.ml.early_case_features import CATEGORICAL_FEATURES, EARLY_NUMERIC_FEATURES
from src.ml.feature_registry import (
    FORBIDDEN_FEATURES,
    feature_registry_frame,
    validate_features,
)
from src.ml.historical_provider_features import HISTORICAL_NUMERIC_FEATURES

FORBIDDEN_PREDICTORS = set(FORBIDDEN_FEATURES)


def leakage_audit_table() -> pd.DataFrame:
    frame = feature_registry_frame()
    table = frame[
        [
            "feature_name",
            "feature_group",
            "description",
            "available_at_prediction_time",
            "prediction_stage",
            "source_table",
            "uses_future_information",
            "safe_for_sla_prediction",
            "safe_for_remaining_time_prediction",
            "safe_for_early_anomaly_detection",
            "safe_for_retrospective_analysis",
            "notes",
        ]
    ].rename(columns={"feature_name": "feature"})
    table["used_in_sla_model"] = table.apply(
        lambda row: (
            "Target only"
            if row["feature"] == "sla_breached"
            else ("Yes" if row["safe_for_sla_prediction"] else "No")
        ),
        axis=1,
    )
    table["available_at_prediction_time"] = table["available_at_prediction_time"].map(
        {True: "Yes", False: "No"}
    )
    return table


def assert_no_forbidden_features(model_features: list[str]) -> None:
    try:
        validate_features(model_features, "sla_prediction")
    except ValueError as error:
        raise ValueError(f"Feature leakage check failed: {error}") from error


def early_model_features() -> list[str]:
    return CATEGORICAL_FEATURES + EARLY_NUMERIC_FEATURES + HISTORICAL_NUMERIC_FEATURES
