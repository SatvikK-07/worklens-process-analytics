import ast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from app.components.page import PLOT_CONFIG, base_layout, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import (
    load_cases,
    load_features,
    load_model_artifact,
    load_model_metrics,
)
from src.ml.explainability import global_feature_importance, local_shap_explanation
from src.ml.feature_engineering import MODEL_FEATURES

apply_enterprise_theme()
page_header(
    "Predictive intervention",
    "SLA breach prediction",
    "Prioritize recall for breached cases, inspect model quality, and explain case-level risk.",
)

cases = load_cases()
features = load_features()
artifact = load_model_artifact("sla_breach_model")
metadata = load_model_metrics("sla_breach_model")
selected_metrics = metadata["models"][metadata["selected_model"]]
operating_threshold = float(artifact["threshold"])
test_window = metadata["split_boundaries"]["test"]

metrics = st.columns(3)
metrics[0].metric("Selected model", metadata["selected_model"])
metrics[1].metric("Recall", f"{selected_metrics['recall']:.1%}")
metrics[2].metric("Precision", f"{selected_metrics['precision']:.1%}")
metrics = st.columns(3)
metrics[0].metric("F1", f"{selected_metrics['f1']:.1%}")
metrics[1].metric("ROC–AUC", f"{selected_metrics['roc_auc']:.3f}")
metrics[2].metric("PR–AUC", f"{selected_metrics['pr_auc']:.3f}")

st.markdown(
    f"""
    <div class="insight-box">
      <small>DECISION POLICY</small>
      <p>The operating threshold is <b>{operating_threshold:.0%}</b>, selected on
      the temporal validation window using F2 to emphasize costly false negatives.
      The displayed metrics come from the later, untouched test window.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

features["created_at"] = pd.to_datetime(features["created_at"])
test_features = features[features["created_at"].between(test_window["start"], test_window["end"])]
probability = artifact["pipeline"].predict_proba(test_features[MODEL_FEATURES])[:, 1]
target = test_features["sla_breached"]
fpr, tpr, _ = roc_curve(target, probability)
precision, recall, _ = precision_recall_curve(target, probability)

left, middle, right = st.columns([1, 1, 1])
with left:
    section_header("Confusion matrix", "Holdout results at the operating threshold.")
    labels = ["On time", "Breached"]
    matrix = selected_metrics["confusion_matrix"]
    figure = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=",d",
        color_continuous_scale=["#edf8f3", "#19a974", "#10231c"],
        labels=dict(x="Predicted", y="Actual", color="Cases"),
    )
    st.plotly_chart(base_layout(figure, 360), width="stretch", config=PLOT_CONFIG)
with middle:
    section_header("ROC curve", "True-positive rate versus false-positive rate.")
    figure = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#19a974", width=3)))
    figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#a8b5af"))
    figure.update_xaxes(title="False-positive rate")
    figure.update_yaxes(title="True-positive rate")
    st.plotly_chart(base_layout(figure, 360), width="stretch", config=PLOT_CONFIG)
with right:
    section_header("Precision–recall", "Tradeoff for the breached-case class.")
    figure = go.Figure(
        go.Scatter(x=recall, y=precision, mode="lines", line=dict(color="#e05d44", width=3))
    )
    figure.update_xaxes(title="Recall")
    figure.update_yaxes(title="Precision")
    st.plotly_chart(base_layout(figure, 360), width="stretch", config=PLOT_CONFIG)

section_header(
    "Validation realism",
    "Temporal evaluation predicts later cases from earlier cases; the random split is shown only as a comparison.",
)
left, right = st.columns(2)
with left:
    comparison = pd.DataFrame(metadata["split_comparison"]).T.reset_index()
    comparison = comparison.rename(columns={"index": "split"})
    st.dataframe(
        comparison[["split", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
        width="stretch",
        hide_index=True,
    )
    boundaries = pd.DataFrame(metadata["split_boundaries"]).T
    st.caption(
        f"Train {boundaries.loc['train', 'rows']:,} · "
        f"Validation {boundaries.loc['validation', 'rows']:,} · "
        f"Test {boundaries.loc['test', 'rows']:,} cases"
    )
with right:
    calibration = metadata["calibration_curve"]
    figure = go.Figure(
        go.Scatter(
            x=calibration["mean_predicted_probability"],
            y=calibration["fraction_positive"],
            mode="lines+markers",
            name="Observed",
            line=dict(color="#19a974", width=3),
        )
    )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash", color="#a8b5af"),
    )
    figure.update_xaxes(title="Mean predicted probability")
    figure.update_yaxes(title="Observed breach rate")
    st.plotly_chart(base_layout(figure, 330), width="stretch", config=PLOT_CONFIG)

section_header(
    "Threshold analysis", "Validation-window precision and recall by operating threshold."
)
thresholds = pd.DataFrame(metadata["threshold_analysis"])
threshold_figure = px.line(
    thresholds,
    x="threshold",
    y=["precision", "recall", "f2"],
    markers=True,
    color_discrete_map={
        "precision": "#295c49",
        "recall": "#e05d44",
        "f2": "#4a70b5",
    },
)
threshold_figure.add_vline(
    x=operating_threshold,
    line_dash="dash",
    line_color="#10231c",
    annotation_text="Selected",
)
threshold_figure.update_yaxes(tickformat=".0%")
st.plotly_chart(base_layout(threshold_figure, 360), width="stretch", config=PLOT_CONFIG)

left, right = st.columns([1, 1.15])
with left:
    section_header("Global risk drivers", "Model feature importance for the selected estimator.")
    importance = global_feature_importance(
        artifact,
        12,
        sample=test_features.sample(min(300, len(test_features)), random_state=42),
    ).sort_values("importance")
    importance["feature"] = (
        importance["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
        .str.replace("_", " ")
    )
    figure = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#c9eee0", "#19a974", "#10231c"],
    )
    figure.update_coloraxes(showscale=False)
    st.plotly_chart(base_layout(figure, 460), width="stretch", config=PLOT_CONFIG)
with right:
    section_header(
        "Case-level risk review", "Select a case to inspect risk drivers and next action."
    )
    case_id = st.selectbox(
        "Case ID",
        cases.sort_values("sla_breach_probability", ascending=False)["case_id"],
        index=0,
    )
    case = cases[cases["case_id"] == case_id].iloc[0]
    risk = float(case["sla_breach_probability"])
    one, two, three = st.columns(3)
    one.metric("Breach risk", f"{risk:.1%}")
    two.metric("Risk level", case["risk_level"])
    three.metric("SLA", f"{case['sla_threshold_hours']:.0f}h")
    try:
        factors = ast.literal_eval(case["top_risk_factors"])
    except (ValueError, SyntaxError):
        factors = [case["top_risk_factors"]]
    st.markdown("**Top drivers**")
    for factor in factors:
        st.markdown(f"- {factor}")
    st.success(f"Recommended action: {case['recommended_action']}")
    feature_row = features[features["case_id"] == case_id].iloc[0]
    shap_detail = local_shap_explanation(
        artifact,
        feature_row,
        background=test_features.sample(min(100, len(test_features)), random_state=7),
        top_n=6,
    ).sort_values("shap_value")
    shap_detail["feature"] = (
        shap_detail["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
        .str.replace("_", " ")
    )
    st.caption("Local SHAP contribution (model log-odds scale)")
    local_figure = px.bar(
        shap_detail,
        x="shap_value",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={
            "Increases risk": "#e05d44",
            "Decreases risk": "#19a974",
        },
    )
    st.plotly_chart(base_layout(local_figure, 320), width="stretch", config=PLOT_CONFIG)

section_header("Highest-risk cases", "Operational queue ordered by predicted breach probability.")
high_risk = cases[cases["sla_breach_probability"] >= 0.5].sort_values(
    "sla_breach_probability", ascending=False
)
st.dataframe(
    high_risk[
        [
            "case_id",
            "claim_type",
            "priority",
            "region",
            "sla_threshold_hours",
            "sla_breach_probability",
            "risk_level",
            "recommended_action",
        ]
    ].head(500),
    width="stretch",
    hide_index=True,
    column_config={
        "sla_breach_probability": st.column_config.ProgressColumn(
            "Breach risk", min_value=0, max_value=1, format="%.0%%"
        ),
        "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
    },
)
