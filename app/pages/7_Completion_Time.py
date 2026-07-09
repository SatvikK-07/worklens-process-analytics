import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_model_metrics

apply_enterprise_theme()
page_header(
    "Duration forecasting",
    "Completion time prediction",
    "Estimate end-to-end completion time and identify cases likely to exceed their SLA.",
)

cases = load_cases().dropna(subset=["predicted_completion_hours"]).copy()
metadata = load_model_metrics("completion_time_model")
selected = metadata["models"][metadata["selected_model"]]
cases["prediction_error"] = cases["predicted_completion_hours"] - cases["total_duration_hours"]
cases["predicted_breach"] = cases["predicted_completion_hours"] > cases["sla_threshold_hours"]

metrics = st.columns(5)
metrics[0].metric("Selected model", metadata["selected_model"])
metrics[1].metric("MAE", f"{selected['mae']:.2f}h")
metrics[2].metric("RMSE", f"{selected['rmse']:.2f}h")
metrics[3].metric("R²", f"{selected['r2']:.3f}")
metrics[4].metric("Likely breaches", f"{cases['predicted_breach'].sum():,}")

left, right = st.columns([1.2, 1])
with left:
    section_header("Predicted versus actual", "A well-calibrated model follows the diagonal.")
    sample = cases.sample(min(8_000, len(cases)), random_state=42)
    figure = px.scatter(
        sample,
        x="total_duration_hours",
        y="predicted_completion_hours",
        color="sla_breached",
        opacity=0.42,
        color_continuous_scale=["#19a974", "#e05d44"],
        hover_data=["case_id", "claim_type", "priority"],
    )
    maximum = max(sample["total_duration_hours"].max(), sample["predicted_completion_hours"].max())
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=maximum,
        y1=maximum,
        line=dict(dash="dash", color="#52665e"),
    )
    st.plotly_chart(base_layout(figure, 500), width="stretch", config=PLOT_CONFIG)
with right:
    section_header("Error distribution", "Prediction error in hours; zero is exact.")
    error_frame = cases.copy()
    error_frame["prediction_error"] = error_frame["prediction_error"].clip(-20, 20)
    figure = px.histogram(
        error_frame,
        x="prediction_error",
        nbins=60,
        color_discrete_sequence=["#295c49"],
    )
    st.plotly_chart(base_layout(figure, 320), width="stretch", config=PLOT_CONFIG)
    section_header("Case estimate")
    case_id = st.selectbox(
        "Case ID",
        cases.sort_values("predicted_completion_hours", ascending=False)["case_id"],
    )
    case = cases[cases["case_id"] == case_id].iloc[0]
    one, two = st.columns(2)
    one.metric("Predicted completion", f"{case['predicted_completion_hours']:.1f}h")
    two.metric("Expected SLA", f"{case['sla_threshold_hours']:.0f}h")
    if case["predicted_breach"]:
        st.error("Likely breach · prioritize before the next constrained review queue.")
    else:
        st.success("Expected to complete within SLA under current process conditions.")

section_header("Longest predicted cases")
st.dataframe(
    cases.sort_values("predicted_completion_hours", ascending=False)[
        [
            "case_id",
            "claim_type",
            "priority",
            "region",
            "predicted_completion_hours",
            "total_duration_hours",
            "sla_threshold_hours",
            "predicted_breach",
        ]
    ].head(500),
    width="stretch",
    hide_index=True,
    column_config={
        "predicted_completion_hours": st.column_config.NumberColumn(
            "Predicted hours", format="%.1f"
        ),
        "total_duration_hours": st.column_config.NumberColumn("Actual hours", format="%.1f"),
        "predicted_breach": st.column_config.CheckboxColumn("Likely breach"),
    },
)
