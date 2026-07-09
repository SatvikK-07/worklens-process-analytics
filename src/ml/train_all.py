from __future__ import annotations

import json
import sqlite3

import pandas as pd

from src.ml.anomaly_detection import anomaly_reason, train_anomaly_model
from src.ml.explainability import case_risk_factors, recommended_action
from src.ml.feature_engineering import build_case_features
from src.ml.retrospective_anomaly_detection import (
    train_retrospective_anomaly_model,
)
from src.ml.train_completion_model import train_completion_models
from src.ml.train_sla_model import train_sla_models
from src.utils.config import settings
from src.utils.db import database_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.25:
        return "Medium"
    return "Low"


def train_all() -> dict[str, dict]:
    cases = pd.read_csv(settings.data_dir / "cases.csv")
    events = pd.read_csv(settings.data_dir / "events.csv")
    providers = pd.read_csv(settings.data_dir / "providers.csv")
    features = build_case_features(cases, events, providers)
    LOGGER.info("Built %s case feature rows", f"{len(features):,}")

    _, sla_probability, sla_metrics = train_sla_models(features, settings.random_seed)
    LOGGER.info("Selected SLA model: %s", sla_metrics["selected_model"])
    _, completion_prediction, completion_metrics = train_completion_models(
        features, settings.random_seed
    )
    LOGGER.info("Selected completion model: %s", completion_metrics["selected_model"])
    _, anomaly_score, anomaly_metrics = train_anomaly_model(features, settings.random_seed)
    _, retrospective, retrospective_metrics = train_retrospective_anomaly_model(
        cases, events, settings.random_seed
    )
    retrospective = retrospective.set_index("case_id")

    predictions = pd.DataFrame(
        {
            "case_id": features["case_id"],
            "sla_breach_probability": sla_probability,
            "predicted_completion_hours": completion_prediction,
            "anomaly_score": anomaly_score,
            "early_anomaly_score": anomaly_score,
            "retrospective_anomaly_score": features["case_id"].map(
                retrospective["retrospective_anomaly_score"]
            ),
            "retrospective_anomaly_flag": features["case_id"].map(
                retrospective["retrospective_anomaly_flag"]
            ),
        }
    )
    predictions["risk_level"] = predictions["sla_breach_probability"].map(risk_level)
    predictions["top_risk_factors"] = [
        json.dumps(case_risk_factors(row)) for _, row in features.iterrows()
    ]
    predictions["recommended_action"] = [
        recommended_action(row, float(probability))
        for (_, row), probability in zip(
            features.iterrows(),
            predictions["sla_breach_probability"],
            strict=True,
        )
    ]
    predictions["anomaly_reason"] = [anomaly_reason(row) for _, row in features.iterrows()]
    predictions.to_csv(settings.data_dir / "predictions.csv", index=False)

    path = database_path()
    if path.exists():
        with sqlite3.connect(path) as connection:
            connection.execute("DELETE FROM predictions")
            database_predictions = predictions.drop(
                columns=[
                    "anomaly_reason",
                    "early_anomaly_score",
                    "retrospective_anomaly_score",
                    "retrospective_anomaly_flag",
                ]
            )
            database_predictions.to_sql("predictions", connection, if_exists="append", index=False)
            connection.commit()
    else:
        LOGGER.info("SQLite database not present; predictions saved to CSV only")

    return {
        "sla": sla_metrics,
        "completion": completion_metrics,
        "anomaly": anomaly_metrics,
        "retrospective_anomaly": retrospective_metrics,
    }


if __name__ == "__main__":
    metrics = train_all()
    LOGGER.info("Training complete: %s", metrics)
