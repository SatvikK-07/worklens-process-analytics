from typing import Any

from fastapi import APIRouter, Depends

from backend.services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/model", tags=["models"])


@router.get("/metrics")
def get_model_metrics(
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    return service.metrics()


@router.get("/metadata")
def get_model_metadata(
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    return service.metrics()


@router.get("/features/leakage-audit")
def get_leakage_audit(
    service: ModelService = Depends(get_model_service),
) -> list[dict[str, Any]]:
    return service.leakage_audit()
