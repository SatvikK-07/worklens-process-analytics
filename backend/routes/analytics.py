from typing import Any

from fastapi import APIRouter, Depends, Query

from backend.services.data_service import DataService, get_data_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/bottlenecks")
def get_bottlenecks(
    limit: int = Query(default=10, ge=1, le=50),
    service: DataService = Depends(get_data_service),
) -> list[dict[str, Any]]:
    return service.bottlenecks(limit)


@router.get("/anomalies")
def get_anomalies(
    limit: int = Query(default=20, ge=1, le=100),
    service: DataService = Depends(get_data_service),
) -> list[dict[str, Any]]:
    return service.anomalies(limit)
