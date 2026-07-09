from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import (
    DurationPrediction,
    EarlyAnomalyPrediction,
    EarlyCaseFeatures,
    RetrospectiveAnomalyPrediction,
    RetrospectiveCaseFeatures,
    SLAPrediction,
)
from backend.services.prediction_service import PredictionService, get_prediction_service

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/sla", response_model=SLAPrediction)
def predict_sla(
    payload: EarlyCaseFeatures,
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    try:
        return service.predict_sla(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are not trained") from exc


@router.post("/duration", response_model=DurationPrediction)
def predict_duration(
    payload: EarlyCaseFeatures,
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    try:
        return service.predict_duration(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are not trained") from exc


@router.post("/anomaly/early", response_model=EarlyAnomalyPrediction)
def predict_early_anomaly(
    payload: EarlyCaseFeatures,
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    try:
        return service.predict_early_anomaly(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are not trained") from exc


@router.post("/anomaly/retrospective", response_model=RetrospectiveAnomalyPrediction)
def predict_retrospective_anomaly(
    payload: RetrospectiveCaseFeatures,
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    try:
        return service.predict_retrospective_anomaly(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are not trained") from exc
