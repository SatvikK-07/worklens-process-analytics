"""Backward-compatible early anomaly API.

New code should import `early_anomaly_detection` or
`retrospective_anomaly_detection` explicitly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.ml.early_anomaly_detection import train_early_anomaly_model
from src.ml.model_registry import save_model


def train_anomaly_model(
    features: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.Series, dict[str, Any]]:
    artifact, scores, metadata = train_early_anomaly_model(features, random_seed)
    save_model("anomaly_model", artifact, metadata)
    return artifact, scores.rename("anomaly_score"), metadata


def anomaly_reason(row: pd.Series) -> str:
    reasons = []
    if row["handoff_count"] >= 4:
        reasons.append(f"{int(row['handoff_count'])} handoffs observed in prefix")
    if row["rework_count"] >= 1:
        reasons.append(f"{int(row['rework_count'])} rework loops observed in prefix")
    if row["provider_clarification_count"] >= 1:
        reasons.append("provider clarification observed")
    if row["queue_wait_time_total"] >= 24:
        reasons.append("high observed queue wait")
    return ", ".join(reasons[:3]) or "rare early routing pattern"
