import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, money, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_bottlenecks, load_cases, load_events

apply_enterprise_theme()
page_header(
    "Flow constraints",
    "Bottleneck analysis",
    "Separate active handling effort from queue delay and rank the constraints that matter.",
)

bottlenecks = load_bottlenecks()
events = load_events()
cases = load_cases()
top = bottlenecks.iloc[0]

metrics = st.columns(5)
metrics[0].metric("Top constraint", top["activity"])
metrics[1].metric("Bottleneck score", f"{top['bottleneck_score']:.0f}/100")
metrics[2].metric("Average wait", f"{top['avg_wait_hours']:.1f}h")
metrics[3].metric("P95 effort", f"{top['p95_duration_minutes']:.0f} min")
metrics[4].metric("Labor exposure", money(top["labor_cost"]))

st.markdown(
    f"""
    <div class="insight-box">
      <small>PRIORITY RECOMMENDATION</small>
      <p>Prioritize <b>{top["activity"]}</b>. It touches {int(top["total_cases"]):,}
      cases and accounts for {top["delay_contribution"]:.1%} of measured queue delay.
      Start with capacity balancing, routing rules, and queue-age alerts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 1])
with left:
    section_header(
        "Constraint ranking", "Composite score across duration, P95, wait, SLA impact, and rework."
    )
    ranking = bottlenecks.sort_values("bottleneck_score")
    figure = px.bar(
        ranking,
        x="bottleneck_score",
        y="activity",
        orientation="h",
        color="avg_wait_hours",
        color_continuous_scale=["#bdebd9", "#f4b860", "#d4513d"],
        hover_data=["avg_duration_minutes", "p95_duration_minutes", "total_cases"],
    )
    st.plotly_chart(base_layout(figure, 500), width="stretch", config=PLOT_CONFIG)

with right:
    section_header("Delay Pareto", "Cumulative share of queue delay by activity.")
    pareto = bottlenecks.sort_values("total_wait_hours", ascending=False).copy()
    pareto["cumulative_delay"] = (
        pareto["total_wait_hours"].cumsum() / pareto["total_wait_hours"].sum()
    )
    figure = px.bar(pareto, x="activity", y="total_wait_hours", color_discrete_sequence=["#295c49"])
    figure.add_scatter(
        x=pareto["activity"],
        y=pareto["cumulative_delay"],
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#e05d44", width=3),
        name="Cumulative share",
    )
    figure.update_layout(
        yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.08]),
    )
    figure.update_xaxes(tickangle=-40)
    st.plotly_chart(base_layout(figure, 500), width="stretch", config=PLOT_CONFIG)

section_header(
    "Duration distribution",
    "Active handling-time variation; outliers indicate inconsistent execution.",
)
activity_order = bottlenecks["activity"].tolist()
sample = events.sample(min(75_000, len(events)), random_state=42)
figure = px.box(
    sample,
    x="activity",
    y="duration_minutes",
    category_orders={"activity": activity_order},
    color="team",
    points=False,
)
figure.update_xaxes(tickangle=-35)
st.plotly_chart(base_layout(figure, 470), width="stretch", config=PLOT_CONFIG)

section_header("Ranked activity detail")
display = bottlenecks.copy()
display["delay_contribution"] *= 100
display["sla_breach_contribution"] *= 100
display["rework_rate"] *= 100
st.dataframe(
    display,
    width="stretch",
    hide_index=True,
    column_config={
        "activity": "Activity",
        "bottleneck_score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.0f"
        ),
        "delay_contribution": st.column_config.NumberColumn("Delay share", format="%.1f%%"),
        "sla_breach_contribution": st.column_config.NumberColumn("SLA impact", format="%.1f%%"),
        "rework_rate": st.column_config.NumberColumn("Rework", format="%.1f%%"),
        "labor_cost": st.column_config.NumberColumn("Labor cost", format="$%.0f"),
    },
)
st.download_button(
    "Download bottleneck analysis",
    display.to_csv(index=False),
    "worklens_bottlenecks.csv",
    "text/csv",
)
