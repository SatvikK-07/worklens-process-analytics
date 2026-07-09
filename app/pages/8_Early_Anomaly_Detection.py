import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_features, load_model_metrics

apply_enterprise_theme()
page_header(
    "Operational assurance",
    "Early anomaly detection",
    "Surface unusual patterns using only the first five observed events and prior history.",
)

cases = load_cases()
features = load_features()
metadata = load_model_metrics("anomaly_model")
threshold = metadata["score_threshold"]
scored = features[["case_id", "handoff_count", "rework_count", "queue_wait_time_total"]].merge(
    cases[["case_id", "claim_type", "priority", "region", "early_anomaly_score"]],
    on="case_id",
)
flagged = scored[scored["early_anomaly_score"] >= threshold].sort_values(
    "early_anomaly_score", ascending=False
)

metrics = st.columns(5)
metrics[0].metric("Scored cases", f"{len(cases):,}")
metrics[1].metric("Cases flagged", f"{len(flagged):,}")
metrics[2].metric("Alert rate", f"{len(flagged) / len(cases):.2%}")
metrics[3].metric("Threshold", f"{threshold:.1%}")
metrics[4].metric("Mode", "Prefix-safe")

left, right = st.columns([1.25, 1])
with left:
    section_header(
        "Anomaly score distribution", "The intervention threshold isolates the extreme tail."
    )
    figure = px.histogram(
        cases,
        x="early_anomaly_score",
        nbins=70,
        color_discrete_sequence=["#295c49"],
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#e05d44",
        annotation_text="Investigation threshold",
    )
    st.plotly_chart(base_layout(figure, 420), width="stretch", config=PLOT_CONFIG)
with right:
    section_header(
        "Observed routing signals",
        "Only handoffs, rework, and queue wait seen in the prediction prefix are shown.",
    )
    figure = px.scatter(
        scored.sample(min(10_000, len(scored)), random_state=42),
        x="queue_wait_time_total",
        y="rework_count",
        color="early_anomaly_score",
        size="handoff_count",
        color_continuous_scale=["#c9eee0", "#f4b860", "#d4513d"],
        hover_data=["case_id", "claim_type", "priority"],
    )
    st.plotly_chart(base_layout(figure, 420), width="stretch", config=PLOT_CONFIG)

section_header(
    "Investigation queue", "Review the highest-scoring cases before expanding the alert threshold."
)
columns = [
    "case_id",
    "claim_type",
    "priority",
    "region",
    "queue_wait_time_total",
    "handoff_count",
    "rework_count",
    "early_anomaly_score",
]
st.dataframe(
    flagged[columns],
    width="stretch",
    hide_index=True,
    column_config={
        "early_anomaly_score": st.column_config.ProgressColumn(
            "Anomaly score", min_value=0, max_value=1, format="%.1%%"
        ),
    },
)
