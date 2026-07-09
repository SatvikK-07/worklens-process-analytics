from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.ml.feature_registry import validate_features
from src.ml.model_registry import save_model
from src.ml.post_completion_features import build_post_completion_features

RETROSPECTIVE_NUMERIC_FEATURES = [
    "total_duration_hours",
    "total_cost",
    "final_event_count",
    "final_unique_teams",
    "final_manual_minutes",
]


def train_retrospective_anomaly_model(
    cases: pd.DataFrame,
    events: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    validate_features(RETROSPECTIVE_NUMERIC_FEATURES, "retrospective_analysis")
    retrospective = build_post_completion_features(cases, events)
    numeric = retrospective[RETROSPECTIVE_NUMERIC_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    scaler = RobustScaler()
    transformed = scaler.fit_transform(numeric)
    model = IsolationForest(
        n_estimators=220,
        contamination=0.015,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(transformed)
    raw_scores = -model.score_samples(transformed)
    retrospective["retrospective_anomaly_score"] = pd.Series(raw_scores).rank(pct=True)
    threshold = float(retrospective["retrospective_anomaly_score"].quantile(0.985))
    retrospective["retrospective_anomaly_flag"] = (
        retrospective["retrospective_anomaly_score"] >= threshold
    ).astype(int)
    artifact = {
        "model": model,
        "scaler": scaler,
        "numeric_features": RETROSPECTIVE_NUMERIC_FEATURES,
        "threshold": threshold,
        "mode": "post_completion",
        "training_raw_scores": np.sort(raw_scores),
    }
    metadata = {
        "model": "Isolation Forest on completed-case features",
        "mode": "retrospective_anomaly_investigation",
        "threshold": threshold,
        "flagged_cases": int(retrospective["retrospective_anomaly_flag"].sum()),
    }
    save_model("retrospective_anomaly_model", artifact, metadata)
    return artifact, retrospective, metadata


def score_retrospective_anomaly(artifact: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    validate_features(artifact["numeric_features"], "retrospective_analysis")
    transformed = artifact["scaler"].transform(features[artifact["numeric_features"]])
    raw_scores = -artifact["model"].score_samples(transformed)
    reference = np.asarray(artifact.get("training_raw_scores", raw_scores))
    return np.searchsorted(reference, raw_scores, side="right") / max(len(reference), 1)
