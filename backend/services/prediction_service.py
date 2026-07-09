from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from backend.schemas import EarlyCaseFeatures, RetrospectiveCaseFeatures
from src.ml.anomaly_detection import anomaly_reason
from src.ml.early_anomaly_detection import score_early_anomaly
from src.ml.explainability import case_risk_factors, recommended_action
from src.ml.model_registry import load_model
from src.ml.retrospective_anomaly_detection import score_retrospective_anomaly


def risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.25:
        return "Medium"
    return "Low"


class PredictionService:
    def __init__(
        self,
        sla_artifact: dict[str, Any],
        duration_artifact: dict[str, Any],
        early_anomaly_artifact: dict[str, Any] | None = None,
        retrospective_anomaly_artifact: dict[str, Any] | None = None,
    ) -> None:
        self.sla_artifact = sla_artifact
        self.duration_artifact = duration_artifact
        self.early_anomaly_artifact = early_anomaly_artifact
        self.retrospective_anomaly_artifact = retrospective_anomaly_artifact

    def predict_sla(self, payload: EarlyCaseFeatures) -> dict[str, Any]:
        row = pd.Series(payload.model_dump())
        frame = pd.DataFrame([row])
        probability = float(
            self.sla_artifact["pipeline"].predict_proba(frame[self.sla_artifact["features"]])[0, 1]
        )
        threshold = float(self.sla_artifact["threshold"])
        return {
            "sla_breach_probability": probability,
            "risk_level": risk_level(probability),
            "operating_threshold": threshold,
            "top_risk_factors": case_risk_factors(row),
            "recommended_action": recommended_action(row, probability),
        }

    def predict_duration(self, payload: EarlyCaseFeatures) -> dict[str, float]:
        frame = pd.DataFrame([payload.model_dump()])
        prediction = float(
            self.duration_artifact["pipeline"].predict(frame[self.duration_artifact["features"]])[0]
        )
        return {"predicted_completion_hours": max(0.0, prediction)}

    def predict_early_anomaly(self, payload: EarlyCaseFeatures) -> dict[str, Any]:
        if self.early_anomaly_artifact is None:
            raise FileNotFoundError("Early anomaly artifact is not loaded")
        frame = pd.DataFrame([payload.model_dump()])
        score = float(score_early_anomaly(self.early_anomaly_artifact, frame)[0])
        threshold = float(self.early_anomaly_artifact["threshold"])
        reason = anomaly_reason(frame.iloc[0])
        return {
            "mode": "early_prefix_risk",
            "anomaly_score": score,
            "flagged": score >= threshold,
            "threshold": threshold,
            "top_associated_drivers": [part.strip() for part in reason.split(",")],
        }

    def predict_retrospective_anomaly(self, payload: RetrospectiveCaseFeatures) -> dict[str, Any]:
        if self.retrospective_anomaly_artifact is None:
            raise FileNotFoundError("Retrospective anomaly artifact is not loaded")
        frame = pd.DataFrame([payload.model_dump()])
        score = float(score_retrospective_anomaly(self.retrospective_anomaly_artifact, frame)[0])
        threshold = float(self.retrospective_anomaly_artifact["threshold"])
        drivers = frame.iloc[0].sort_values(ascending=False).head(3).index.tolist()
        return {
            "mode": "retrospective_investigation",
            "anomaly_score": score,
            "flagged": score >= threshold,
            "threshold": threshold,
            "top_associated_drivers": [driver.replace("_", " ") for driver in drivers],
        }


@lru_cache(maxsize=1)
def _load_prediction_service() -> PredictionService:
    return PredictionService(
        sla_artifact=load_model("sla_breach_model"),
        duration_artifact=load_model("completion_time_model"),
        early_anomaly_artifact=load_model("anomaly_model"),
        retrospective_anomaly_artifact=load_model("retrospective_anomaly_model"),
    )


class LazyPredictionService:
    """Delay artifact loading until after FastAPI validates the request body."""

    def predict_sla(self, payload: EarlyCaseFeatures) -> dict[str, Any]:
        return _load_prediction_service().predict_sla(payload)

    def predict_duration(self, payload: EarlyCaseFeatures) -> dict[str, float]:
        return _load_prediction_service().predict_duration(payload)

    def predict_early_anomaly(self, payload: EarlyCaseFeatures) -> dict[str, Any]:
        return _load_prediction_service().predict_early_anomaly(payload)

    def predict_retrospective_anomaly(self, payload: RetrospectiveCaseFeatures) -> dict[str, Any]:
        return _load_prediction_service().predict_retrospective_anomaly(payload)


def get_prediction_service() -> LazyPredictionService:
    return LazyPredictionService()
