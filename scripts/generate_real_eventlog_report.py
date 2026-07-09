from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_eventlog_experiments.scripts.run_all_real_experiments import (
    _generate_figures,
)

RESULTS = ROOT / "real_eventlog_experiments" / "results"
REPORT = ROOT / "reports" / "real_eventlog_model_report.md"


def _load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `make run-real-eventlog-sample` or "
            "`make run-real-eventlog-full` first."
        )
    return json.loads(path.read_text())


def _display(value: float | None, digits: int = 4) -> str:
    return "not defined" if value is None else f"{value:.{digits}f}"


def _model_table(models: dict[str, dict[str, Any]], classification: bool) -> str:
    if classification:
        columns = [
            "model",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "brier_score",
        ]
    else:
        columns = ["model", "mae", "rmse", "median_absolute_error", "r2"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for model, metrics in models.items():
        values = [model, *[_display(metrics.get(column)) for column in columns[1:]]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate() -> Path:
    summary = _load("experiment_summary.json")
    classification = _load("long_case_n3_metrics.json")
    regression = _load("remaining_time_n3_metrics.json")
    selected_classification = classification["models"][classification["selected_model"]]
    selected_regression = regression["models"][regression["selected_model"]]
    median_baseline = regression["models"]["Median Baseline"]
    beats_regression_baseline = selected_regression["mae"] < median_baseline["mae"]
    classification_lift = selected_classification["pr_auc_lift_over_prevalence"]
    variant_lines = [
        "| target | selected model | prevalence | ROC-AUC | PR-AUC | PR lift |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target, values in classification["target_variants"].items():
        metrics = values["test_metrics"]
        variant_lines.append(
            "| "
            + " | ".join(
                [
                    target,
                    values["selected_model"],
                    _display(metrics["positive_prevalence"]),
                    _display(metrics["roc_auc"]),
                    _display(metrics["pr_auc"]),
                    _display(metrics["pr_auc_lift_over_prevalence"]),
                ]
            )
            + " |"
        )

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(
        f"""# Real event-log model report

Generated from machine-readable results by `scripts/generate_real_eventlog_report.py`.

## Dataset

- Dataset: {summary["dataset"]}
- Run type: `{summary["run_type"]}`
- Cases/events: {summary["case_count"]:,} / {summary["event_count"]:,}
- Activities/variants: {summary["activity_count"]:,} / {summary["variant_count"]:,}
- Date range: {summary["start_timestamp"]} to {summary["end_timestamp"]}
- Validation: chronological 70/15/15 primary; random split diagnostic only

## Classification task

- Prediction moment: after the first {classification["prefix_length"]} events
- Validation-selected target: `{classification["selected_target"]}`
- Definition: {classification["target_definition"]}
- Selected model: {classification["selected_model"]}
- Selection used validation metrics only

{_model_table(classification["models"], classification=True)}

### Target variants

{chr(10).join(variant_lines)}

The primary temporal test PR-AUC lift over prevalence is
{classification_lift:.3f}. A value above 1 beats the majority/prevalence ranking
baseline; a value at or below 1 does not. Target-variant test results are
diagnostic and were not used to change the validation-selected primary task.

## Remaining-time regression

- Prediction moment: after the first {regression["prefix_length"]} events
- Target: remaining hours after the prefix timestamp
- Non-baseline target transform: log1p during fitting, expm1 after prediction
- Validation-selected model: {regression["selected_model"]}

{_model_table(regression["models"], classification=False)}

Selected MAE: {selected_regression["mae"]:.2f} hours. Median-baseline MAE:
{median_baseline["mae"]:.2f} hours. **Beats baseline:
{"yes" if beats_regression_baseline else "no"}**.

## Leakage controls

- Prefix fields contain only the first N observed events.
- Duration, remaining time, final event count, end timestamp, and future events
  are forbidden model inputs.
- Historical rate/duration encodings are fitted on the training window only.
- Fitting rows use leave-one-out encodings.
- Unseen validation/test categories use training-window defaults.

## Interpretation

The full experiment is evidence for reproducible, leakage-controlled predictive
monitoring—not clinical usefulness. Temporal performance remains weak and
unstable across target definitions. The richer feature layer and log target
reduce remaining-time MAE substantially relative to the earlier implementation,
but the report retains the median baseline whenever it is better.

## Limitations and next work

The public log is hospital care, not claims operations. Prefix paths are sparse,
duration is heavy-tailed, and resource/context fields are limited. Next work
would use rolling-origin backtests, censor-aware survival methods, richer
resource/case attributes, and calibration monitoring on a governed enterprise
event stream.
"""
    )
    _generate_figures(RESULTS, ROOT / "reports")
    print(f"Generated {REPORT.relative_to(ROOT)} from saved real-log results")
    return REPORT


if __name__ == "__main__":
    generate()
