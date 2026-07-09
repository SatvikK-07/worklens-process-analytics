import math

import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_automation_candidates

apply_enterprise_theme()
page_header(
    "Automation portfolio",
    "Automation opportunities",
    "Prioritize work using volume, effort, repeatability, feasibility, error, and modeled savings.",
)

candidates = load_automation_candidates()
monthly_total = candidates["estimated_monthly_savings"].sum()
annual_total = candidates["estimated_annual_savings"].sum()

metrics = st.columns(4)
metrics[0].metric("Modeled monthly value", money(monthly_total))
metrics[1].metric("Annual value", money(annual_total))
metrics[2].metric("Top candidate", candidates.iloc[0]["activity"])
metrics[3].metric("Priority score", f"{candidates.iloc[0]['automation_priority_score']:.0f}/100")

left, right = st.columns([1.25, 1])
with left:
    section_header(
        "Opportunity frontier", "Value and feasibility; bubble size represents monthly volume."
    )
    figure = px.scatter(
        candidates,
        x="automation_feasibility",
        y="estimated_monthly_savings",
        size="monthly_volume",
        color="automation_priority_score",
        text="activity",
        color_continuous_scale=["#c9eee0", "#19a974", "#10231c"],
        size_max=48,
    )
    figure.update_traces(textposition="top center")
    figure.update_xaxes(tickformat=".0%", range=[0, 1.05])
    figure.update_yaxes(tickprefix="$")
    st.plotly_chart(base_layout(figure, 500), width="stretch", config=PLOT_CONFIG)

with right:
    section_header("What-if calculator", "Adjust assumptions for a candidate business case.")
    activity = st.selectbox("Candidate", candidates["activity"])
    selected = candidates[candidates["activity"] == activity].iloc[0]
    hourly_cost = st.number_input("Hourly labor cost", 10, 120, 35, 1, format="%d")
    coverage = st.slider(
        "Automatable share",
        0,
        100,
        int(selected["automation_feasibility"] * 100),
        5,
    )
    implementation_cost = st.number_input("Implementation cost", 5_000, 500_000, 65_000, 5_000)
    savings = (
        selected["monthly_volume"]
        * selected["avg_duration_minutes"]
        / 60
        * hourly_cost
        * coverage
        / 100
    )
    payback = implementation_cost / max(savings, 1)
    one, two = st.columns(2)
    one.metric("Monthly savings", money(savings))
    two.metric("Payback", f"{payback:.1f} months")
    st.progress(min(int(coverage), 100), text=f"{coverage}% activity coverage")
    st.caption(
        f"Year-one net value: {money(savings * 12 - implementation_cost)} after implementation cost."
    )

section_header(
    "Recommended roadmap", "Sequenced by priority score with manageable portfolio waves."
)
roadmap = candidates.head(8).copy()
roadmap["phase"] = [f"Phase {math.ceil((index + 1) / 2)}" for index in range(len(roadmap))]
for phase, group in roadmap.groupby("phase", sort=False):
    with st.expander(f"{phase} · {' + '.join(group['activity'])}", expanded=phase == "Phase 1"):
        st.write(
            f"Combined modeled annual savings: **{money(group['estimated_annual_savings'].sum())}**"
        )
        st.dataframe(
            group[
                [
                    "activity",
                    "monthly_volume",
                    "avg_duration_minutes",
                    "automation_feasibility",
                    "estimated_monthly_savings",
                    "automation_priority_score",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

section_header("Candidate scoring detail")
st.dataframe(
    candidates,
    width="stretch",
    hide_index=True,
    column_config={
        "automation_priority_score": st.column_config.ProgressColumn(
            "Priority", min_value=0, max_value=100, format="%.0f"
        ),
        "automation_feasibility": st.column_config.NumberColumn("Feasibility", format="%.0%%"),
        "estimated_monthly_savings": st.column_config.NumberColumn(
            "Monthly savings", format="$%.0f"
        ),
        "estimated_annual_savings": st.column_config.NumberColumn("Annual savings", format="$%.0f"),
    },
)
