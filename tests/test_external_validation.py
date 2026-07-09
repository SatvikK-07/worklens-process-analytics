from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from src.external_validation.sepsis import (
    external_validation_report,
    file_md5,
    parse_sepsis_xes,
)

TINY_XES = b"""<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="Case-1"/>
    <event>
      <string key="concept:name" value="Register"/>
      <date key="time:timestamp" value="2025-01-01T00:00:00+00:00"/>
      <string key="org:group" value="Intake"/>
    </event>
    <event>
      <string key="concept:name" value="Review"/>
      <date key="time:timestamp" value="2025-01-01T01:00:00+00:00"/>
      <string key="org:group" value="Review"/>
    </event>
  </trace>
  <trace>
    <string key="concept:name" value="Case-2"/>
    <event>
      <string key="concept:name" value="Register"/>
      <date key="time:timestamp" value="2025-01-02T00:00:00+00:00"/>
      <string key="org:group" value="Intake"/>
    </event>
    <event>
      <string key="concept:name" value="Review"/>
      <date key="time:timestamp" value="2025-01-02T01:00:00+00:00"/>
      <string key="org:group" value="Review"/>
    </event>
  </trace>
</log>"""


def test_xes_parser_reads_cases_and_events(tmp_path: Path) -> None:
    path = tmp_path / "tiny.xes.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(TINY_XES)
    events = parse_sepsis_xes(path)
    assert len(events) == 4
    assert events["case_id"].nunique() == 2
    assert events["timestamp"].dt.tz is not None


def test_file_md5_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("worklens")
    assert file_md5(path) == "9350fc797552fbdb3c7f643ed5f1d48a"


def test_external_report_runs_process_algorithms() -> None:
    sample = pd.read_csv("data/sample/sepsis_events_sample.csv")
    sample["timestamp"] = pd.to_datetime(sample["timestamp"], utc=True)
    report = external_validation_report(sample)
    assert report["case_count"] == 35
    assert report["algorithm_checks"]["transition_analysis"]
    assert report["algorithm_checks"]["chronological_order"]


def test_public_sample_has_required_event_columns() -> None:
    sample = pd.read_csv("data/sample/sepsis_events_sample.csv")
    assert {"case_id", "activity", "timestamp", "event_id"}.issubset(sample.columns)
    assert len(sample) > 400
