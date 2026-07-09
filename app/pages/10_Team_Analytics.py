import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_events
from src.analytics.bottleneck_engine import calculate_wait_times

apply_enterprise_theme()
page_header(
    "Capacity intelligence",
    "Team analytics",
    "Understand process variation, workload distribution, and queue pressure at team level.",
)
st.caption(
    "This view is designed for capacity planning and process improvement—not individual surveillance."
)

cases = load_cases()
events = calculate_wait_times(load_events())
events = events.merge(
    cases[["case_id", "total_duration_hours", "sla_breached", "rework_count"]],
    on="case_id",
    how="left",
)
team = events.groupby("team", as_index=False).agg(
    cases_handled=("case_id", "nunique"),
    events_completed=("event_id", "count"),
    avg_activity_minutes=("duration_minutes", "mean"),
    median_activity_minutes=("duration_minutes", "median"),
    avg_queue_hours=("wait_hours", "mean"),
    sla_breach_rate=("sla_breached", "mean"),
    rework_rate=("rework_count", lambda values: (values > 0).mean()),
)
team["productivity_index"] = 100 * (
    team["events_completed"] / team["events_completed"].max() * 0.55
    + (1 - team["avg_queue_hours"] / team["avg_queue_hours"].max()) * 0.45
)

metrics = st.columns(4)
metrics[0].metric("Operational teams", f"{len(team)}")
metrics[1].metric("Highest queue pressure", team.loc[team["avg_queue_hours"].idxmax(), "team"])
metrics[2].metric("Most workload", team.loc[team["events_completed"].idxmax(), "team"])
metrics[3].metric("Median queue wait", f"{team['avg_queue_hours'].median():.1f}h")

left, right = st.columns(2)
with left:
    section_header("Workload distribution", "Event volume and cases touched by team.")
    figure = px.bar(
        team.sort_values("events_completed"),
        x="events_completed",
        y="team",
        orientation="h",
        color="cases_handled",
        color_continuous_scale=["#c9eee0", "#19a974", "#10231c"],
    )
    st.plotly_chart(base_layout(figure, 430), width="stretch", config=PLOT_CONFIG)
with right:
    section_header("Queue pressure", "Average wait before activities owned by each team.")
    figure = px.scatter(
        team,
        x="avg_queue_hours",
        y="sla_breach_rate",
        size="events_completed",
        color="rework_rate",
        text="team",
        color_continuous_scale=["#c9eee0", "#f4b860", "#d4513d"],
    )
    figure.update_traces(textposition="top center")
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(base_layout(figure, 430), width="stretch", config=PLOT_CONFIG)

section_header("Team-level process variation")
display = team.copy()
display["sla_breach_rate"] *= 100
display["rework_rate"] *= 100
st.dataframe(
    display.sort_values("avg_queue_hours", ascending=False),
    width="stretch",
    hide_index=True,
    column_config={
        "productivity_index": st.column_config.ProgressColumn(
            "Capacity index", min_value=0, max_value=100, format="%.0f"
        ),
        "sla_breach_rate": st.column_config.NumberColumn("SLA breach", format="%.1f%%"),
        "rework_rate": st.column_config.NumberColumn("Rework", format="%.1f%%"),
        "avg_queue_hours": st.column_config.NumberColumn("Avg queue h", format="%.1f"),
    },
)
