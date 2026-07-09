from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import CaseDetail, CaseExplanation
from backend.services.data_service import DataService, get_data_service

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("/{case_id}", response_model=CaseDetail)
def get_case(
    case_id: str,
    service: DataService = Depends(get_data_service),
) -> dict:
    detail = service.case_detail(case_id.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return detail


@router.get("/{case_id}/explanation", response_model=CaseExplanation)
def get_case_explanation(
    case_id: str,
    service: DataService = Depends(get_data_service),
) -> dict:
    explanation = service.case_explanation(case_id.upper())
    if explanation is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return explanation
