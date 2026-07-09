from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.ml.feature_engineering import NUMERIC_FEATURES
from src.ml.feature_registry import validate_features


def train_early_anomaly_model(
    features: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.Series, dict[str, Any]]:
    validate_features(NUMERIC_FEATURES, "early_anomaly_detection")
    numeric = features[NUMERIC_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
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
    statistical_score = pd.Series(raw_scores).rank(pct=True)
    rule_score = (
        (features["handoff_count"] >= 4).astype(float)
        + (features["rework_count"] >= 1).astype(float)
        + (features["queue_wait_time_total"] >= 24).astype(float)
    ) / 3
    combined = pd.Series(
        np.clip(0.75 * statistical_score.to_numpy() + 0.25 * rule_score, 0, 1),
        index=features.index,
        name="early_anomaly_score",
    )
    threshold = float(combined.quantile(0.985))
    artifact = {
        "model": model,
        "scaler": scaler,
        "numeric_features": NUMERIC_FEATURES,
        "threshold": threshold,
        "mode": "early_prefix",
        "training_raw_scores": np.sort(raw_scores),
    }
    metadata = {
        "model": "Isolation Forest + prefix-safe operational rules",
        "mode": "early_anomaly_risk",
        "contamination": 0.015,
        "threshold": threshold,
        "score_threshold": threshold,
        "flagged_cases": int((combined >= threshold).sum()),
    }
    return artifact, combined, metadata


def score_early_anomaly(artifact: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    validate_features(artifact["numeric_features"], "early_anomaly_detection")
    transformed = artifact["scaler"].transform(features[artifact["numeric_features"]])
    raw_scores = -artifact["model"].score_samples(transformed)
    reference = np.asarray(artifact.get("training_raw_scores", raw_scores))
    percentile = np.searchsorted(reference, raw_scores, side="right") / max(len(reference), 1)
    rule_score = (
        (features["handoff_count"] >= 4).astype(float)
        + (features["rework_count"] >= 1).astype(float)
        + (features["queue_wait_time_total"] >= 24).astype(float)
    ) / 3
    return np.clip(0.75 * percentile + 0.25 * rule_score.to_numpy(), 0, 1)
