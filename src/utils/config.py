from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = Path(os.getenv("WORKLENS_DATA_DIR", ROOT_DIR / "data" / "synthetic"))
    model_dir: Path = Path(os.getenv("WORKLENS_MODEL_DIR", ROOT_DIR / "models"))
    database_url: str = os.getenv("WORKLENS_DATABASE_URL", f"sqlite:///{ROOT_DIR / 'worklens.db'}")
    random_seed: int = int(os.getenv("WORKLENS_RANDOM_SEED", "42"))
    hourly_labor_cost: float = float(os.getenv("WORKLENS_HOURLY_LABOR_COST", "35"))
    log_level: str = os.getenv("WORKLENS_LOG_LEVEL", "INFO")


settings = Settings()

SLA_THRESHOLDS = {
    "Medical Claim": 72,
    "Pharmacy Claim": 48,
    "Prior Authorization": 48,
    "Appeal": 120,
    "High-Cost Specialty Drug": 72,
    "Out-of-Network Claim": 96,
}

ALLOWED_ACTIVITIES = (
    "Claim Received",
    "Eligibility Check",
    "Member Info Correction",
    "Document Intake",
    "Document Review",
    "Provider Clarification",
    "Medical Necessity Review",
    "Nurse Review",
    "Medical Director Review",
    "Approval",
    "Denial",
    "Payment Processing",
    "Case Closed",
)
