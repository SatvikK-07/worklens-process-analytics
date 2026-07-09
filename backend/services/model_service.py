from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.ml.feature_registry import feature_registry_frame
from src.ml.model_registry import load_metadata

MODEL_NAMES = [
    "sla_breach_model",
    "completion_time_model",
    "anomaly_model",
    "retrospective_anomaly_model",
]


class ModelService:
    def metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in MODEL_NAMES:
            try:
                result[name] = load_metadata(name)
            except FileNotFoundError:
                result[name] = {"status": "artifact_missing"}
        return {
            "data_tracks": {
                "synthetic": "Product demonstration only",
                "real_event_log": "Independent modelling validation",
            },
            "models": result,
        }

    def leakage_audit(self) -> list[dict[str, Any]]:
        frame = feature_registry_frame().astype(object)
        return frame.where(frame.notna(), None).to_dict("records")


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()
