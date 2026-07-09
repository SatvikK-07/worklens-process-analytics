from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.external_validation.sepsis import parse_sepsis_xes

STANDARD_COLUMNS = ["case_id", "activity", "timestamp", "resource", "lifecycle"]


def normalize_event_log(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {"team": "resource"}
    normalized = frame.rename(columns=rename).copy()
    for column in ("resource", "lifecycle"):
        if column not in normalized:
            normalized[column] = "Unknown"
    missing = {"case_id", "activity", "timestamp"} - set(normalized.columns)
    if missing:
        raise ValueError(f"Event log missing required columns: {sorted(missing)}")
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    validate_event_log(normalized)
    return normalized.sort_values(
        ["case_id", "timestamp", "event_id"]
        if "event_id" in normalized
        else ["case_id", "timestamp"]
    ).reset_index(drop=True)


def validate_event_log(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Event log is empty")
    if frame["case_id"].isna().any():
        raise ValueError("case_id cannot be null")
    if frame["activity"].isna().any():
        raise ValueError("activity cannot be null")
    if pd.to_datetime(frame["timestamp"], errors="coerce").isna().any():
        raise ValueError("timestamp contains unparseable values")
    duplicate_columns = (
        ["event_id"] if "event_id" in frame else ["case_id", "activity", "timestamp"]
    )
    if frame.duplicated(duplicate_columns).any():
        raise ValueError("duplicate event rows detected")


def load_eventlog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See real_eventlog_experiments/README.md.")
    if path.name.endswith(".xes.gz"):
        frame = parse_sepsis_xes(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Supported formats are .csv and .xes.gz")
    return normalize_event_log(frame)
