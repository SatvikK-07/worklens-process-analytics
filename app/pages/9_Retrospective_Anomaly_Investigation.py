import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_model_metrics

apply_enterprise_theme()
page_header(
    "Completed-case assurance",
    "Retrospective anomaly investigation",
    "Investigate unusual completed cases using final duration, cost, and full-path evidence.",
)

cases = load_cases().dropna(subset=["retrospective_anomaly_score"]).copy()
metadata = load_model_metrics("retrospective_anomaly_model")
threshold = float(metadata["threshold"])
flagged = cases[cases["retrospective_anomaly_score"] >= threshold].sort_values(
    "retrospective_anomaly_score", ascending=False
)

st.warning(
    "This mode uses post-completion fields and must not be interpreted as an early prediction."
)
metrics = st.columns(4)
metrics[0].metric("Completed cases scored", f"{len(cases):,}")
metrics[1].metric("Cases flagged", f"{len(flagged):,}")
metrics[2].metric("Investigation threshold", f"{threshold:.1%}")
metrics[3].metric("Flagged cost exposure", money(flagged["total_cost"].sum()))

left, right = st.columns(2)
with left:
    section_header("Completed-case score distribution")
    figure = px.histogram(
        cases,
        x="retrospective_anomaly_score",
        nbins=60,
        color_discrete_sequence=["#295c49"],
    )
    figure.add_vline(x=threshold, line_dash="dash", line_color="#e05d44")
    st.plotly_chart(base_layout(figure, 420), width="stretch", config=PLOT_CONFIG)
with right:
    section_header("Duration and cost outliers")
    figure = px.scatter(
        cases.sample(min(10_000, len(cases)), random_state=42),
        x="total_duration_hours",
        y="total_cost",
        color="retrospective_anomaly_score",
        size="rework_count",
        hover_data=["case_id", "claim_type", "priority"],
        color_continuous_scale=["#c9eee0", "#f4b860", "#d4513d"],
    )
    st.plotly_chart(base_layout(figure, 420), width="stretch", config=PLOT_CONFIG)

section_header("Completed-case investigation queue")
st.dataframe(
    flagged[
        [
            "case_id",
            "claim_type",
            "priority",
            "total_duration_hours",
            "total_cost",
            "rework_count",
            "retrospective_anomaly_score",
        ]
    ],
    width="stretch",
    hide_index=True,
)
