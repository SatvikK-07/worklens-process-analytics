import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_rework_loops

apply_enterprise_theme()
page_header(
    "Avoidable work",
    "Rework loops",
    "Find repeated activity cycles, estimate their cost, and connect each pattern to a root-cause hypothesis.",
)

cases = load_cases()
loops = load_rework_loops()
total_cost = (loops["avg_extra_cost"] * loops["occurrences"]).sum()
total_hours = (loops["avg_extra_time_hours"] * loops["occurrences"]).sum()
top = loops.iloc[0]

metrics = st.columns(5)
metrics[0].metric("Cases with rework", f"{(cases['rework_count'] > 0).sum():,}")
metrics[1].metric("Rework rate", f"{(cases['rework_count'] > 0).mean():.1%}")
metrics[2].metric("Detected patterns", f"{len(loops):,}")
metrics[3].metric("Extra elapsed time", f"{total_hours:,.0f}h")
metrics[4].metric("Direct rework cost", money(total_cost))

st.markdown(
    f"""
    <div class="insight-box">
      <small>ROOT-CAUSE SIGNAL</small>
      <p><b>{top["loop_pattern"]}</b> is the most prevalent loop, affecting
      {int(top["cases_affected"]):,} cases. Likely cause:
      <b>{top["root_cause_hypothesis"]}</b>. Add pre-submission validation and
      route incomplete cases before formal review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 1])
with left:
    section_header("Pattern frequency", "Cases affected by each detected A → B → A loop.")
    chart = loops.head(10).sort_values("cases_affected")
    figure = px.bar(
        chart,
        x="cases_affected",
        y="loop_pattern",
        orientation="h",
        color="avg_extra_time_hours",
        color_continuous_scale=["#bdebd9", "#f4b860", "#d4513d"],
    )
    st.plotly_chart(base_layout(figure, 430), width="stretch", config=PLOT_CONFIG)
with right:
    section_header("Rework trend", "Monthly share of cases containing repeated activity.")
    monthly = cases.assign(month=cases["created_at"].dt.to_period("M").dt.to_timestamp())
    monthly = (
        monthly.groupby("month", as_index=False)["rework_count"]
        .agg(lambda values: (values > 0).mean())
        .rename(columns={"rework_count": "rework_rate"})
    )
    figure = px.area(
        monthly,
        x="month",
        y="rework_rate",
        markers=True,
        color_discrete_sequence=["#19a974"],
    )
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(base_layout(figure, 430), width="stretch", config=PLOT_CONFIG)

section_header("Ranked loop detail")
st.dataframe(
    loops,
    width="stretch",
    hide_index=True,
    column_config={
        "loop_pattern": st.column_config.TextColumn("Loop pattern", width="large"),
        "occurrences": st.column_config.NumberColumn("Occurrences", format="%d"),
        "cases_affected": st.column_config.NumberColumn("Cases", format="%d"),
        "avg_extra_time_hours": st.column_config.NumberColumn("Extra hours", format="%.1f"),
        "avg_extra_cost": st.column_config.NumberColumn("Extra cost", format="$%.0f"),
        "root_cause_hypothesis": st.column_config.TextColumn("Likely root cause", width="large"),
    },
)
