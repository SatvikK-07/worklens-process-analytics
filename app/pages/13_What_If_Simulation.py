import plotly.graph_objects as go
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header
from app.components.styles import apply_enterprise_theme
from app.data import (
    load_automation_candidates,
    load_bottlenecks,
    load_cases,
    load_rework_loops,
)
from src.analytics.simulation_engine import simulate_operating_model

apply_enterprise_theme()
page_header(
    "Scenario planning",
    "What-if simulation",
    "Model how automation, capacity, and rework interventions could change operating outcomes.",
)

cases = load_cases()
candidates = load_automation_candidates()
bottlenecks = load_bottlenecks()
loops = load_rework_loops()

left, right = st.columns([1, 1.4])
with left:
    st.markdown("### Intervention levers")
    activity = st.selectbox("Automate activity", candidates["activity"])
    candidate = candidates[candidates["activity"] == activity].iloc[0]
    automation_coverage = st.slider("Automation coverage", 0, 100, 70, 5) / 100
    capacity_increase = st.slider("Medical director capacity increase", 0, 60, 20, 5) / 100
    rework_reduction = st.slider("Provider clarification loop reduction", 0, 80, 30, 5) / 100
    implementation_cost = st.number_input("Implementation cost", 10_000, 1_000_000, 120_000, 10_000)

baseline_avg = float(cases["total_duration_hours"].mean())
baseline_breach = float(cases["sla_breached"].mean())
activity_hours_per_case = (
    candidate["frequency"] * candidate["avg_duration_minutes"] / 60 / len(cases)
)
md_delay_share = float(
    bottlenecks.loc[
        bottlenecks["activity"] == "Medical Director Review", "delay_contribution"
    ].iloc[0]
)
rework_hours_per_case = float(
    (loops["avg_extra_time_hours"] * loops["occurrences"]).sum() / len(cases)
)
scenario = simulate_operating_model(
    baseline_avg,
    baseline_breach,
    activity_hours_per_case,
    automation_coverage,
    md_delay_share,
    capacity_increase,
    rework_hours_per_case,
    rework_reduction,
)
monthly_savings = candidate["estimated_monthly_savings"] * automation_coverage
payback = implementation_cost / max(monthly_savings, 1)

with right:
    st.markdown("### Modeled outcome")
    metrics = st.columns(3)
    metrics[0].metric(
        "Average handling time",
        f"{scenario['new_avg_hours']:.1f}h",
        f"−{scenario['hours_reduced']:.1f}h",
        delta_color="normal",
    )
    metrics[1].metric(
        "SLA breach rate",
        f"{scenario['new_breach_rate']:.1%}",
        f"−{scenario['breach_rate_reduction']:.1%}",
        delta_color="normal",
    )
    metrics[2].metric("Payback", f"{payback:.1f} months")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Baseline",
            x=["Avg handling hours", "SLA breach %"],
            y=[baseline_avg, baseline_breach * 100],
            marker_color="#9caaa4",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Scenario",
            x=["Avg handling hours", "SLA breach %"],
            y=[scenario["new_avg_hours"], scenario["new_breach_rate"] * 100],
            marker_color="#19a974",
        )
    )
    figure.update_layout(barmode="group")
    st.plotly_chart(base_layout(figure, 390), width="stretch", config=PLOT_CONFIG)
    st.success(
        f"Modeled monthly savings: {money(monthly_savings)} · "
        f"Year-one net value: {money(monthly_savings * 12 - implementation_cost)}"
    )
    st.caption(
        "Scenario estimates are directional and assume intervention effects are independent. "
        "Validate with a controlled pilot before committing capital."
    )
