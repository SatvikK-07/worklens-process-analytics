import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import (
    load_automation_candidates,
    load_bottlenecks,
    load_cases,
)
from src.analytics.kpi_engine import monthly_trends

apply_enterprise_theme()
page_header(
    "Command center",
    "Executive summary",
    "A decision-ready view of throughput, service risk, avoidable work, and automation value.",
)

cases = load_cases()
bottlenecks = load_bottlenecks()
candidates = load_automation_candidates()
trends = monthly_trends(cases)
last_month = trends.iloc[-1]
previous_month = trends.iloc[-2]


def delta(current: float, previous: float, inverse: bool = False) -> str:
    change = 100 * (current - previous) / max(abs(previous), 1e-9)
    prefix = "↓" if change < 0 else "↑"
    return f"{prefix} {abs(change):.1f}% vs prior month"


top_bottleneck = bottlenecks.iloc[0]
monthly_savings = candidates["estimated_monthly_savings"].sum()

row_one = st.columns(4)
row_one[0].metric(
    "Total cases",
    f"{len(cases):,}",
    delta(last_month["cases_processed"], previous_month["cases_processed"]),
)
row_one[1].metric(
    "Completed",
    f"{cases['closed_at'].notna().sum():,}",
    f"{cases['closed_at'].notna().mean():.1%} of volume",
)
row_one[2].metric("Open cases", f"{cases['closed_at'].isna().sum():,}")
row_one[3].metric(
    "Avg handling time",
    f"{cases['total_duration_hours'].mean():.1f}h",
    delta(last_month["avg_handling_hours"], previous_month["avg_handling_hours"]),
    delta_color="inverse",
)
row_two = st.columns(4)
row_two[0].metric("Median handling", f"{cases['total_duration_hours'].median():.1f}h")
row_two[1].metric(
    "SLA breach rate",
    f"{cases['sla_breached'].mean():.1%}",
    delta(last_month["sla_breach_rate"], previous_month["sla_breach_rate"]),
    delta_color="inverse",
)

row_two[2].metric(
    "Rework rate",
    f"{(cases['rework_count'] > 0).mean():.1%}",
    delta(last_month["rework_rate"], previous_month["rework_rate"]),
    delta_color="inverse",
)
row_two[3].metric("Monthly automation value", money(monthly_savings))
row_three = st.columns(3)
row_three[0].metric("Top bottleneck", top_bottleneck["activity"])
row_three[1].metric(
    "Critical-risk cases",
    f"{cases['sla_breach_probability'].ge(0.75).sum():,}",
)
row_three[2].metric("Anomalous cases", f"{cases['anomaly_label'].sum():,}")

st.markdown(
    f"""
    <div class="insight-box">
      <small>EXECUTIVE SIGNAL</small>
      <p><b>{top_bottleneck["activity"]}</b> is the highest-scoring bottleneck at
      {top_bottleneck["bottleneck_score"]:.0f}/100. WorkLens identified
      <b>{money(monthly_savings)}</b> in modeled monthly automation value, led by
      <b>{candidates.iloc[0]["activity"]}</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.65, 1])
with left:
    section_header("Operating trend", "Monthly volume and service-level performance.")
    trend_long = trends.melt(
        id_vars="month",
        value_vars=["sla_breach_rate", "rework_rate"],
        var_name="metric",
        value_name="rate",
    )
    trend_long["metric"] = trend_long["metric"].map(
        {"sla_breach_rate": "SLA breach", "rework_rate": "Rework"}
    )
    figure = px.line(
        trend_long,
        x="month",
        y="rate",
        color="metric",
        markers=True,
        color_discrete_map={"SLA breach": "#e05d44", "Rework": "#19a974"},
    )
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(base_layout(figure), width="stretch", config=PLOT_CONFIG)

with right:
    section_header("Cases processed", "Monthly intake volume.")
    volume = px.bar(
        trends,
        x="month",
        y="cases_processed",
        color="cases_processed",
        color_continuous_scale=["#dff6ec", "#19a974", "#10231c"],
    )
    volume.update_coloraxes(showscale=False)
    st.plotly_chart(base_layout(volume), width="stretch", config=PLOT_CONFIG)

left, right = st.columns(2)
with left:
    section_header("Top delay constraints", "Composite bottleneck score by activity.")
    top = bottlenecks.head(6).sort_values("bottleneck_score")
    figure = px.bar(
        top,
        x="bottleneck_score",
        y="activity",
        orientation="h",
        text="bottleneck_score",
        color="bottleneck_score",
        color_continuous_scale=["#f5c76a", "#e05d44", "#8f2d25"],
    )
    figure.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    figure.update_coloraxes(showscale=False)
    st.plotly_chart(base_layout(figure), width="stretch", config=PLOT_CONFIG)

with right:
    section_header("Automation value", "Estimated monthly savings by candidate.")
    top = candidates.head(6).sort_values("estimated_monthly_savings")
    figure = px.bar(
        top,
        x="estimated_monthly_savings",
        y="activity",
        orientation="h",
        text="estimated_monthly_savings",
        color="automation_priority_score",
        color_continuous_scale=["#a7e6ce", "#19a974", "#10231c"],
    )
    figure.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    figure.update_coloraxes(showscale=False)
    st.plotly_chart(base_layout(figure), width="stretch", config=PLOT_CONFIG)

with st.expander("Review monthly operating detail"):
    detail = trends.copy()
    detail["month"] = detail["month"].dt.strftime("%b %Y")
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            "sla_breach_rate": st.column_config.NumberColumn(format="%.1%%"),
            "rework_rate": st.column_config.NumberColumn(format="%.1%%"),
            "cost_leakage": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
