from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RealEvidence:
    summary: dict[str, Any] | None
    classification: dict[str, Any] | None
    regression: dict[str, Any] | None
    external_validation: dict[str, Any] | None
    missing: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def load_real_evidence(root: Path) -> RealEvidence:
    results = root / "real_eventlog_experiments" / "results"
    paths = {
        "experiment summary": results / "experiment_summary.json",
        "N=3 classification metrics": results / "long_case_n3_metrics.json",
        "N=3 regression metrics": results / "remaining_time_n3_metrics.json",
        "process validation": root / "reports" / "external_validation.json",
    }
    loaded = {label: _read_json(path) for label, path in paths.items()}
    return RealEvidence(
        summary=loaded["experiment summary"],
        classification=loaded["N=3 classification metrics"],
        regression=loaded["N=3 regression metrics"],
        external_validation=loaded["process validation"],
        missing=tuple(label for label, value in loaded.items() if value is None),
    )


def leakage_feature_table(
    used_features: list[str] | None = None,
) -> pd.DataFrame:
    used = set(used_features or [])
    rows = [
        ("first_activity", True, "Observed in the prefix", ""),
        ("elapsed_hours", True, "Elapsed by the prediction timestamp", ""),
        (
            "observed_activity_counts",
            True,
            "Counts only activities already observed",
            "Represented by prefix activity/repeat/count features",
        ),
        (
            "last_activity_train_long_case_rate",
            True,
            "Fitted on train only; leave-one-out on fitting rows",
            "Smoothed historical encoding",
        ),
        (
            "prefix_path_train_long_case_rate_smoothed",
            True,
            "Fitted on train only; unseen paths use training prevalence",
            "Smoothed historical encoding",
        ),
        (
            "total_duration_hours",
            False,
            "Known only after completion",
            "Target/analysis only",
        ),
        (
            "remaining_time_hours",
            False,
            "Prediction target",
            "Target only",
        ),
        ("final_event_count", False, "Requires the complete case", "Forbidden"),
        ("future_activity", False, "Occurs after prediction", "Forbidden"),
    ]
    return pd.DataFrame(
        [
            {
                "feature_name": name,
                "safe_for_prefix_prediction": safe,
                "reason": reason,
                "used_in_model": name in used,
                "notes": notes,
            }
            for name, safe, reason, notes in rows
        ]
    )


def regression_interpretation(regression: dict[str, Any] | None) -> str:
    if not regression:
        return "Regression results are unavailable; run the sample or full experiment."
    selected = regression["models"][regression["selected_model"]]
    baseline = regression["models"]["Median Baseline"]
    if selected["mae"] < baseline["mae"]:
        return (
            f"The selected model beats the temporal median baseline by "
            f"{selected.get('baseline_improvement_pct', 0):.1f}% MAE."
        )
    return (
        f"The selected model does not beat the temporal median baseline "
        f"({selected['mae']:.1f}h versus {baseline['mae']:.1f}h MAE). "
        "The result is retained as negative evidence."
    )
