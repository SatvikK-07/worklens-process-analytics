from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from src.analytics.bottleneck_engine import rank_bottlenecks
from src.utils.config import settings


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe = frame.astype(object).where(pd.notna(frame), None)
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            safe[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return safe.to_dict("records")


class DataService:
    def __init__(
        self,
        cases: pd.DataFrame,
        events: pd.DataFrame,
    ) -> None:
        self.cases = cases
        self.events = events

    def case_detail(self, case_id: str) -> dict[str, Any] | None:
        case = self.cases[self.cases["case_id"] == case_id]
        if case.empty:
            return None
        events = self.events[self.events["case_id"] == case_id].sort_values("timestamp")
        return {"case": _records(case)[0], "events": _records(events)}

    def case_explanation(self, case_id: str) -> dict[str, Any] | None:
        case = self.cases[self.cases["case_id"] == case_id]
        if case.empty:
            return None
        row = _records(case)[0]
        drivers = row.get("top_risk_factors") or []
        if isinstance(drivers, str):
            try:
                drivers = json.loads(drivers)
            except json.JSONDecodeError:
                drivers = [drivers]
        return {
            "case_id": case_id,
            "explanation_type": "Top associated risk drivers; not causal effects",
            "top_associated_risk_drivers": drivers,
            "recommended_action": row.get("recommended_action"),
        }

    def bottlenecks(self, limit: int) -> list[dict[str, Any]]:
        ranked = rank_bottlenecks(self.events, self.cases).head(limit)
        return _records(ranked)

    def anomalies(self, limit: int) -> list[dict[str, Any]]:
        if "anomaly_score" not in self.cases:
            return []
        ranked = self.cases.sort_values("anomaly_score", ascending=False).head(limit)
        columns = [
            "case_id",
            "claim_type",
            "priority",
            "total_duration_hours",
            "anomaly_score",
        ]
        return _records(ranked[columns])


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    cases = pd.read_csv(settings.data_dir / "cases.csv", parse_dates=["created_at", "closed_at"])
    predictions_path = settings.data_dir / "predictions.csv"
    if predictions_path.exists():
        cases = cases.merge(pd.read_csv(predictions_path), on="case_id", how="left")
    events = pd.read_csv(settings.data_dir / "events.csv", parse_dates=["timestamp"])
    return DataService(cases=cases, events=events)
