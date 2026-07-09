import ast

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_events

apply_enterprise_theme()
page_header(
    "Case investigation",
    "Case explorer",
    "Reconstruct a case timeline, explain its risk, and identify the next operational action.",
)

cases = load_cases()
events = load_events()
search = st.text_input("Search case ID", placeholder="C-0000001").strip().upper()
if search and search in set(cases["case_id"]):
    st.session_state["selected_case"] = search
default = st.session_state.get(
    "selected_case",
    cases.sort_values("sla_breach_probability", ascending=False).iloc[0]["case_id"],
)
case_id = st.selectbox(
    "Case",
    cases["case_id"],
    index=int(cases.index[cases["case_id"] == default][0]),
    label_visibility="collapsed",
)
case = cases[cases["case_id"] == case_id].iloc[0]
timeline = events[events["case_id"] == case_id].sort_values("timestamp").copy()
timeline["end"] = timeline["timestamp"] + pd.to_timedelta(timeline["duration_minutes"], unit="m")

metrics = st.columns(3)
metrics[0].metric("Status", "Closed" if pd.notna(case["closed_at"]) else "Open")
metrics[1].metric("Duration", f"{case['total_duration_hours']:.1f}h")
metrics[2].metric("SLA", "Breached" if case["sla_breached"] else "Met")
metrics = st.columns(3)
metrics[0].metric("Rework", f"{case['rework_count']:.0f}")
metrics[1].metric("Cost", money(case["total_cost"]))
metrics[2].metric("Predicted risk", f"{case['sla_breach_probability']:.1%}")

left, right = st.columns([1.6, 1])
with left:
    section_header("Event timeline", "Active handling blocks shown in chronological order.")
    figure = px.timeline(
        timeline,
        x_start="timestamp",
        x_end="end",
        y="activity",
        color="team",
        hover_data=["user_id", "application_used", "duration_minutes", "status"],
    )
    figure.update_yaxes(autorange="reversed")
    st.plotly_chart(base_layout(figure, 500), width="stretch", config=PLOT_CONFIG)
with right:
    section_header("Case profile")
    st.markdown(
        f"""
        **{case["claim_type"]}** · {case["priority"]} priority  
        {case["region"]} · {case["diagnosis_group"]}  
        Outcome: **{case["outcome"]}**  
        Assigned teams: **{timeline["team"].nunique()}**  
        Applications touched: **{timeline["application_used"].nunique()}**
        """
    )
    section_header("Risk explanation")
    try:
        factors = ast.literal_eval(case["top_risk_factors"])
    except (ValueError, SyntaxError):
        factors = [str(case["top_risk_factors"])]
    for factor in factors:
        st.markdown(f"- {factor}")
    st.success(case["recommended_action"])

section_header("Process path")
st.code(" → ".join(timeline["activity"]), language=None, wrap_lines=True)

section_header("Event detail")
st.dataframe(
    timeline[
        [
            "timestamp",
            "activity",
            "duration_minutes",
            "team",
            "user_id",
            "application_used",
            "status",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "timestamp": st.column_config.DatetimeColumn("Time", format="MMM D, HH:mm"),
        "duration_minutes": st.column_config.NumberColumn("Active min", format="%.1f"),
    },
)
