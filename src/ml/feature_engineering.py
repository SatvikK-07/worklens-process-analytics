from __future__ import annotations

import pandas as pd

from src.ml.early_case_features import (
    CATEGORICAL_FEATURES,
    EARLY_NUMERIC_FEATURES,
    build_early_case_features,
)
from src.ml.historical_provider_features import (
    HISTORICAL_NUMERIC_FEATURES,
    add_historical_features,
)
from src.ml.leakage_audit import assert_no_forbidden_features

NUMERIC_FEATURES = EARLY_NUMERIC_FEATURES + HISTORICAL_NUMERIC_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
assert_no_forbidden_features(MODEL_FEATURES)


def build_case_features(
    cases: pd.DataFrame,
    events: pd.DataFrame,
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """Build leakage-safe first-five-event and as-of historical features."""
    early = build_early_case_features(cases, events, providers)
    features = add_historical_features(early, cases, events)
    targets = cases[
        [
            "case_id",
            "closed_at",
            "total_duration_hours",
            "sla_breached",
            "anomaly_label",
        ]
    ]
    return features.merge(targets, on="case_id", how="left")
