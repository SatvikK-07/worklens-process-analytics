from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EarlyCaseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_type: str
    priority: str
    region: str
    provider_type: str
    diagnosis_group: str
    procedure_group: str
    current_activity: str
    current_team: str
    event_count: int = Field(ge=1, le=5)
    handoff_count: int = Field(ge=0)
    unique_teams_count: int = Field(ge=1)
    rework_count: int = Field(ge=0)
    provider_clarification_count: int = Field(ge=0)
    document_review_duration: float = Field(ge=0)
    medical_review_duration: float = Field(ge=0)
    queue_wait_time_total: float = Field(ge=0)
    elapsed_time_so_far: float = Field(ge=0)
    previous_activity_duration: float = Field(ge=0)
    avg_activity_duration: float = Field(ge=0)
    max_activity_duration: float = Field(ge=0)
    missing_document_flag: int = Field(ge=0, le=1)
    historical_provider_delay_rate: float = Field(ge=0, le=1)
    historical_provider_rework_rate: float = Field(ge=0, le=1)
    provider_history_case_count: int = Field(ge=0)
    historical_team_avg_duration: float = Field(ge=0)
    team_history_case_count: int = Field(ge=0)


class SLAPrediction(BaseModel):
    sla_breach_probability: float
    risk_level: str
    operating_threshold: float
    top_risk_factors: list[str]
    recommended_action: str


class DurationPrediction(BaseModel):
    predicted_completion_hours: float


class EarlyAnomalyPrediction(BaseModel):
    mode: str
    anomaly_score: float = Field(ge=0, le=1)
    flagged: bool
    threshold: float = Field(ge=0, le=1)
    top_associated_drivers: list[str]


class RetrospectiveCaseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_duration_hours: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    final_event_count: int = Field(ge=1)
    final_unique_teams: int = Field(ge=1)
    final_manual_minutes: float = Field(ge=0)


class RetrospectiveAnomalyPrediction(BaseModel):
    mode: str
    anomaly_score: float = Field(ge=0, le=1)
    flagged: bool
    threshold: float = Field(ge=0, le=1)
    top_associated_drivers: list[str]


class CaseDetail(BaseModel):
    case: dict[str, Any]
    events: list[dict[str, Any]]


class CaseExplanation(BaseModel):
    case_id: str
    explanation_type: str
    top_associated_risk_drivers: list[str]
    recommended_action: str | None = None
