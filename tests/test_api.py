from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import EarlyCaseFeatures
from backend.services.data_service import DataService, get_data_service
from backend.services.model_service import get_model_service
from backend.services.prediction_service import (
    PredictionService,
    get_prediction_service,
    risk_level,
)
from src.ml.feature_engineering import MODEL_FEATURES


class FakePipeline:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile([0.18, 0.82], (len(frame), 1))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 54.5)


def payload() -> dict:
    return {
        "claim_type": "Prior Authorization",
        "priority": "High",
        "region": "West",
        "provider_type": "Hospital",
        "diagnosis_group": "Cardiology",
        "procedure_group": "Diagnostic",
        "current_activity": "Document Review",
        "current_team": "Claims Adjudication",
        "event_count": 5,
        "handoff_count": 3,
        "unique_teams_count": 3,
        "rework_count": 1,
        "provider_clarification_count": 1,
        "document_review_duration": 35,
        "medical_review_duration": 0,
        "queue_wait_time_total": 18,
        "elapsed_time_so_far": 21,
        "previous_activity_duration": 35,
        "avg_activity_duration": 18,
        "max_activity_duration": 35,
        "missing_document_flag": 1,
        "historical_provider_delay_rate": 0.28,
        "historical_provider_rework_rate": 0.22,
        "provider_history_case_count": 40,
        "historical_team_avg_duration": 52,
        "team_history_case_count": 100,
    }


def prediction_service() -> PredictionService:
    artifact = {"pipeline": FakePipeline(), "features": MODEL_FEATURES, "threshold": 0.5}
    return PredictionService(artifact, artifact)


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_docs_expose_major_routes() -> None:
    client = TestClient(app)
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/predict/sla" in paths
    assert "/model/features/leakage-audit" in paths


def test_prediction_request_rejects_unknown_or_missing_fields() -> None:
    client = TestClient(app)
    assert client.post("/predict/sla", json={}).status_code == 422
    invalid = payload() | {"final_total_duration": 200}
    assert client.post("/predict/sla", json=invalid).status_code == 422


def test_sla_prediction_endpoint_uses_guarded_service() -> None:
    app.dependency_overrides[get_prediction_service] = prediction_service
    try:
        response = TestClient(app).post("/predict/sla", json=payload())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["risk_level"] == "Critical"
    assert response.json()["sla_breach_probability"] == 0.82


def test_duration_prediction_endpoint() -> None:
    app.dependency_overrides[get_prediction_service] = prediction_service
    try:
        response = TestClient(app).post("/predict/duration", json=payload())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["predicted_completion_hours"] == 54.5


def test_prediction_service_returns_local_explanation() -> None:
    result = prediction_service().predict_sla(EarlyCaseFeatures(**payload()))
    assert "Missing or inconsistent documents" in result["top_risk_factors"]
    assert "Escalate" in result["recommended_action"]


def test_risk_level_boundaries() -> None:
    assert risk_level(0.80) == "Critical"
    assert risk_level(0.60) == "High"
    assert risk_level(0.30) == "Medium"
    assert risk_level(0.10) == "Low"


def data_service() -> DataService:
    cases = pd.DataFrame(
        {
            "case_id": ["C-1", "C-2"],
            "claim_type": ["Medical Claim", "Appeal"],
            "priority": ["Standard", "High"],
            "total_duration_hours": [20.0, 200.0],
            "anomaly_score": [0.1, 0.99],
            "top_risk_factors": ['["High queue wait"]', '["Observed rework"]'],
            "recommended_action": ["Monitor", "Review now"],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["E-1", "E-2", "E-3", "E-4"],
            "case_id": ["C-1", "C-1", "C-2", "C-2"],
            "activity": ["Start", "End", "Start", "End"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-03"]),
            "duration_minutes": [1.0, 2.0, 1.0, 2.0],
            "team": ["A", "B", "A", "B"],
        }
    )
    return DataService(cases, events)


def test_case_endpoint_and_not_found() -> None:
    app.dependency_overrides[get_data_service] = data_service
    try:
        client = TestClient(app)
        found = client.get("/cases/c-1")
        missing = client.get("/cases/nope")
    finally:
        app.dependency_overrides.clear()
    assert found.status_code == 200
    assert len(found.json()["events"]) == 2
    assert missing.status_code == 404


def test_anomaly_endpoint_applies_limit() -> None:
    app.dependency_overrides[get_data_service] = data_service
    try:
        response = TestClient(app).get("/analytics/anomalies?limit=1")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()[0]["case_id"] == "C-2"


def test_data_service_returns_none_for_unknown_case() -> None:
    assert data_service().case_detail("UNKNOWN") is None


class ExpandedPredictionService:
    def predict_early_anomaly(self, _payload) -> dict:
        return {
            "mode": "early_prefix_risk",
            "anomaly_score": 0.91,
            "flagged": True,
            "threshold": 0.88,
            "top_associated_drivers": ["high observed queue wait"],
        }

    def predict_retrospective_anomaly(self, _payload) -> dict:
        return {
            "mode": "retrospective_investigation",
            "anomaly_score": 0.99,
            "flagged": True,
            "threshold": 0.985,
            "top_associated_drivers": ["total duration hours"],
        }


def test_anomaly_prediction_endpoints_enforce_distinct_schemas() -> None:
    app.dependency_overrides[get_prediction_service] = ExpandedPredictionService
    try:
        client = TestClient(app)
        early = client.post("/predict/anomaly/early", json=payload())
        retrospective_payload = {
            "total_duration_hours": 120,
            "total_cost": 800,
            "final_event_count": 20,
            "final_unique_teams": 4,
            "final_manual_minutes": 180,
        }
        retrospective = client.post("/predict/anomaly/retrospective", json=retrospective_payload)
        invalid = client.post(
            "/predict/anomaly/early",
            json=payload() | {"total_duration_hours": 120},
        )
    finally:
        app.dependency_overrides.clear()
    assert early.status_code == 200
    assert early.json()["mode"] == "early_prefix_risk"
    assert retrospective.status_code == 200
    assert retrospective.json()["mode"] == "retrospective_investigation"
    assert invalid.status_code == 422


def test_case_explanation_endpoint_labels_association() -> None:
    app.dependency_overrides[get_data_service] = data_service
    try:
        response = TestClient(app).get("/cases/c-2/explanation")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert "not causal" in response.json()["explanation_type"]
    assert response.json()["top_associated_risk_drivers"] == ["Observed rework"]


class StubModelService:
    def metrics(self) -> dict:
        return {"data_tracks": {"synthetic": "Product demonstration only"}, "models": {}}

    def leakage_audit(self) -> list[dict]:
        return [{"feature_name": "total_cost", "uses_future_information": True}]


def test_model_metrics_and_leakage_endpoints() -> None:
    app.dependency_overrides[get_model_service] = StubModelService
    try:
        client = TestClient(app)
        metrics = client.get("/model/metrics")
        metadata = client.get("/model/metadata")
        audit = client.get("/model/features/leakage-audit")
    finally:
        app.dependency_overrides.clear()
    assert metrics.status_code == 200
    assert metadata.status_code == 200
    assert metrics.json()["data_tracks"]["synthetic"] == "Product demonstration only"
    assert audit.json()[0]["uses_future_information"] is True


class MissingArtifactService:
    def predict_sla(self, _payload):
        raise FileNotFoundError


def test_missing_model_artifact_returns_service_unavailable() -> None:
    app.dependency_overrides[get_prediction_service] = MissingArtifactService
    try:
        response = TestClient(app).post("/predict/sla", json=payload())
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
