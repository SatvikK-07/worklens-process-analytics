from __future__ import annotations

import gzip
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from src.process_mining.path_analysis import get_top_process_paths
from src.process_mining.transition_matrix import calculate_transition_probabilities

DATASET_NAME = "Sepsis Cases - Event Log"
DATASET_DOI = "10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460"
DATASET_PAGE = "https://data.4tu.nl/articles/_/12707639/1"
DOWNLOAD_URL = (
    "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/"
    "643dccf2-985a-459e-835c-a82bce1c0339"
)
EXPECTED_MD5 = "b5671166ac71eb20680d3c74616c43d2"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sepsis_xes(path: Path) -> pd.DataFrame:
    """Parse the public XES log with the standard library."""
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag != "trace":
                continue
            case_id = next(
                (
                    child.attrib.get("value")
                    for child in element
                    if child.tag == "string" and child.attrib.get("key") == "concept:name"
                ),
                None,
            )
            if case_id is None:
                element.clear()
                continue
            for index, event in enumerate(element.findall("event"), start=1):
                attributes = {child.attrib.get("key"): child.attrib.get("value") for child in event}
                rows.append(
                    {
                        "event_id": f"SEPSIS-{case_id}-{index:03d}",
                        "case_id": str(case_id),
                        "activity": attributes.get("concept:name", "Unknown"),
                        "timestamp": attributes.get("time:timestamp"),
                        "team": attributes.get("org:group", "Unknown"),
                        "lifecycle": attributes.get("lifecycle:transition", "complete"),
                        "duration_minutes": 0.0,
                    }
                )
            element.clear()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values(["case_id", "timestamp", "event_id"]).reset_index(drop=True)


def external_validation_report(events: pd.DataFrame) -> dict[str, Any]:
    paths = (
        events.groupby("case_id")["activity"].agg(lambda values: " → ".join(values)).rename("path")
    )
    repeated = events.groupby("case_id")["activity"].agg(
        lambda values: len(values) > len(set(values))
    )
    transitions = calculate_transition_probabilities(events)
    top_paths = get_top_process_paths(events, n=5)
    top_transition = transitions.iloc[0]
    return {
        "dataset": DATASET_NAME,
        "doi": DATASET_DOI,
        "source": DATASET_PAGE,
        "case_count": int(events["case_id"].nunique()),
        "event_count": int(len(events)),
        "activity_count": int(events["activity"].nunique()),
        "process_variant_count": int(paths.nunique()),
        "rework_case_rate": round(float(repeated.mean()), 4),
        "median_events_per_case": float(events.groupby("case_id").size().median()),
        "start_timestamp": events["timestamp"].min().isoformat(),
        "end_timestamp": events["timestamp"].max().isoformat(),
        "top_transition": {
            "source": str(top_transition["source"]),
            "target": str(top_transition["target"]),
            "frequency": int(top_transition["frequency"]),
            "probability": round(float(top_transition["probability"]), 4),
        },
        "top_paths": top_paths.to_dict("records"),
        "algorithm_checks": {
            "transition_analysis": not transitions.empty,
            "path_analysis": not top_paths.empty,
            "chronological_order": bool(
                events.groupby("case_id")["timestamp"]
                .apply(lambda values: values.is_monotonic_increasing)
                .all()
            ),
        },
    }
