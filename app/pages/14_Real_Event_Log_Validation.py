from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.page import page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.real_eventlog_evidence import (
    leakage_feature_table,
    load_real_evidence,
    regression_interpretation,
)

ROOT = Path(__file__).resolve().parents[2]
evidence = load_real_evidence(ROOT)

apply_enterprise_theme()
page_header(
    "Independent methodology evidence",
    "Real event-log validation",
    (
        "Validate process-mining and predictive-monitoring methods on a public "
        "hospital event log—not synthetic claims performance."
    ),
    track="Real public event log — modelling validation",
)
st.warning(
    "This evidence is healthcare-adjacent but not claims-processing validation. "
    "It makes no clinical-validity, causal, or production-readiness claim."
)

if evidence.missing:
    st.info("Some evidence artifacts are unavailable: " + ", ".join(evidence.missing))
    st.code(
        "make run-real-eventlog-sample\n"
        "# Full log, after `make external-data`:\n"
        "make run-real-eventlog-full",
        language="bash",
    )

summary = evidence.summary or evidence.external_validation or {}

section_header(
    "Dataset card",
    "Public source and observed data properties; sample runs are labelled separately.",
)
cards = st.columns(4)
cards[0].metric("Dataset", summary.get("dataset", "Sepsis Cases Event Log"))
cards[1].metric("Cases", f"{summary.get('case_count', 0):,}")
cards[2].metric("Events", f"{summary.get('event_count', 0):,}")
cards[3].metric("Run type", summary.get("run_type", "metrics unavailable"))
cards = st.columns(4)
cards[0].metric("Activities", f"{summary.get('activity_count', 0):,}")
cards[1].metric(
    "Variants",
    f"{summary.get('variant_count', summary.get('process_variant_count', 0)):,}",
)
cards[2].metric(
    "Average events/case",
    f"{summary.get('average_events_per_case', 0):.1f}",
)
cards[3].metric(
    "Median events/case",
    f"{summary.get('median_events_per_case', 0):.1f}",
)
if summary.get("start_timestamp"):
    st.caption(f"Observed date range: {summary['start_timestamp']} to {summary['end_timestamp']}")
st.markdown(
    "Source: [4TU.ResearchData — Sepsis Cases Event Log]"
    "(https://data.4tu.nl/articles/_/12707639/1), DOI "
    "`10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460`."
)
missing_values = summary.get("missing_values", {})
if missing_values:
    st.caption(
        "Missing values: "
        + ", ".join(f"{column}={count}" for column, count in missing_values.items())
    )

section_header("What is validated")
left, right = st.columns(2)
with left:
    st.markdown(
        """
**Process-mining validation**

- XES parsing and normalized event schema
- chronology and deterministic ordering
- transition extraction
- path/variant extraction
- observed repeat/rework analysis
"""
    )
with right:
    st.markdown(
        """
**Predictive-monitoring validation**

- first-N-event prediction snapshots
- chronological train/validation/test split
- train-only historical encodings
- baseline and model comparison
- calibration, threshold, residual, and error analysis
"""
    )

section_header("Process-mining validation")
process = summary.get("process_validation", {})
process_cards = st.columns(4)
process_cards[0].metric("Chronology", "Pass" if process.get("chronology_check") else "Unavailable")
process_cards[1].metric(
    "Transitions", "Pass" if process.get("transition_analysis") else "Unavailable"
)
process_cards[2].metric("Variants", "Pass" if process.get("variant_extraction") else "Unavailable")
process_cards[3].metric("Observed rework", f"{process.get('rework_case_rate', 0):.1%}")
left, right = st.columns(2)
with left:
    st.markdown("**Top observed variants**")
    variants = pd.DataFrame(process.get("top_variants", []))
    if variants.empty:
        st.info("Generate experiment results to display variants.")
    else:
        st.dataframe(variants, width="stretch", hide_index=True)
with right:
    st.markdown("**Top observed transitions**")
    transitions = pd.DataFrame(process.get("top_transitions", []))
    if transitions.empty:
        st.info("Generate experiment results to display transitions.")
    else:
        st.dataframe(transitions, width="stretch", hide_index=True)

section_header(
    "Prediction task A — long/open case classification",
    "Prediction occurs immediately after event N; N=3 is the primary displayed experiment.",
)
classification = evidence.classification
if classification:
    st.markdown(
        f"""
- **Validation-selected target:** `{classification["selected_target"]}`
- **Definition:** {classification["target_definition"]}
- **Temporal split:** earliest 70% train, next 15% validation, latest 15% test
- **Selection rule:** target and estimator selected on validation only
- **Baseline:** majority prior
- **Models:** logistic regression, random forest, histogram gradient boosting
"""
    )
    metrics = classification["models"][classification["selected_model"]]
    values = st.columns(7)
    values[0].metric("Model", classification["selected_model"])
    values[1].metric("ROC–AUC", f"{metrics['roc_auc']:.3f}")
    values[2].metric("PR–AUC", f"{metrics['pr_auc']:.3f}")
    values[3].metric("Precision", f"{metrics['precision']:.1%}")
    values[4].metric("Recall", f"{metrics['recall']:.1%}")
    values[5].metric("F1", f"{metrics['f1']:.3f}")
    values[6].metric("Brier", f"{metrics['brier_score']:.3f}")
    variant_rows = []
    for target, result in classification["target_variants"].items():
        variant_metrics = result["test_metrics"]
        variant_rows.append(
            {
                "target": target,
                "selected_model": result["selected_model"],
                "prevalence": variant_metrics["positive_prevalence"],
                "roc_auc": variant_metrics["roc_auc"],
                "pr_auc": variant_metrics["pr_auc"],
                "pr_lift": variant_metrics["pr_auc_lift_over_prevalence"],
                "balanced_accuracy": variant_metrics["balanced_accuracy"],
            }
        )
    st.dataframe(pd.DataFrame(variant_rows), width="stretch", hide_index=True)
else:
    st.info("Classification metrics are unavailable.")

section_header(
    "Prediction task B — remaining-time regression",
    "Target is hours remaining after event N; non-baseline models use log1p/expm1.",
)
regression = evidence.regression
if regression:
    selected = regression["models"][regression["selected_model"]]
    baseline = regression["models"]["Median Baseline"]
    st.markdown(
        f"""
- **Selection:** lowest chronological validation MAE
- **Baseline:** training-window median remaining time
- **Models:** mean/median baselines, log1p ridge, random forest, robust histogram boosting
- **Selected:** {regression["selected_model"]}
"""
    )
    values = st.columns(6)
    values[0].metric("MAE", f"{selected['mae']:.1f}h")
    values[1].metric("Baseline MAE", f"{baseline['mae']:.1f}h")
    values[2].metric("RMSE", f"{selected['rmse']:.1f}h")
    values[3].metric("Median AE", f"{selected['median_absolute_error']:.1f}h")
    values[4].metric("R²", f"{selected['r2']:.3f}")
    values[5].metric(
        "Baseline improvement",
        f"{selected.get('baseline_improvement_pct', 0):.1f}%",
    )
    st.warning(regression_interpretation(regression))
    st.dataframe(
        pd.DataFrame(regression["error_by_duration_bucket"]),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("Regression metrics are unavailable.")

section_header(
    "Leakage-safe feature audit",
    "Targets and completed-case fields remain in evaluation tables but never enter model input.",
)
used_features = classification.get("features", []) if classification else []
st.dataframe(
    leakage_feature_table(used_features),
    width="stretch",
    hide_index=True,
)
st.caption(
    "Historical duration/rate encodings are fit on train only, use leave-one-out "
    "values on fitting rows, and apply training defaults to unseen validation/test categories."
)

section_header("Performance interpretation")
if classification:
    selected_metrics = classification["models"][classification["selected_model"]]
    lift = selected_metrics["pr_auc_lift_over_prevalence"]
    if lift > 1:
        st.info(
            f"The primary classifier's PR-AUC is {lift:.2f}× prevalence, but "
            "threshold metrics and temporal stability still determine usefulness."
        )
    else:
        st.warning(
            "The primary classifier does not beat prevalence on PR-AUC in the "
            "later holdout. Secondary target variants are shown for diagnosis, "
            "not selected using test results."
        )
st.markdown(
    """
Weak performance is plausible because the prefix is short, 846 variants make
paths sparse, case durations are heavy-tailed, resources are limited, and the
latest time window differs from earlier cases. The useful result is the
measured boundary: richer features and log targets improve robustness but do
not erase temporal drift. Next work would use rolling backtests, censor-aware
survival models, stronger resource/context attributes, and enterprise-specific
event semantics.
"""
)

section_header("Honest conclusion")
validated, not_validated = st.columns(2)
with validated:
    st.success(
        "**Validated**\n\n"
        "- event-log parsing\n"
        "- prefix-safe feature construction\n"
        "- train-only encodings\n"
        "- chronological validation\n"
        "- baseline comparison\n"
        "- leakage guards\n"
        "- process-mining utilities"
    )
with not_validated:
    st.error(
        "**Not validated**\n\n"
        "- real claims-SLA accuracy\n"
        "- clinical decision-making\n"
        "- production deployment\n"
        "- causal inference\n"
        "- audited operational savings"
    )

section_header("Reproduce this evidence")
st.code(
    "# Fast code-path check using the committed 35-case sample\n"
    "make run-real-eventlog-sample\n\n"
    "# Full checksum-verified public log\n"
    "make external-data\n"
    "make run-real-eventlog-full",
    language="bash",
)
