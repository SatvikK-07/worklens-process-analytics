import plotly.express as px
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_events, load_users
from src.analytics.data_quality_engine import run_data_quality_checks

apply_enterprise_theme()
page_header(
    "Trust layer",
    "Data quality",
    "Validate event completeness, chronology, controlled values, references, and process integrity.",
)

cases = load_cases()
events = load_events()
users = load_users()
score, issues = run_data_quality_checks(cases, events, users)

metrics = st.columns(5)
metrics[0].metric("Quality score", f"{score:.1f}/100")
metrics[1].metric("Events checked", f"{len(events):,}")
metrics[2].metric("Cases checked", f"{len(cases):,}")
metrics[3].metric("Issue types", f"{len(issues)}")
metrics[4].metric(
    "Affected records",
    f"{issues['affected_records'].sum():,}" if not issues.empty else "0",
)

if score >= 99:
    st.success(
        "The event log is suitable for production analytics with only isolated seeded exceptions."
    )
elif score >= 95:
    st.warning("The event log is usable, but high-severity exceptions should be resolved.")
else:
    st.error("Resolve the highlighted integrity issues before relying on downstream metrics.")

left, right = st.columns([1.2, 1])
with left:
    section_header("Issues by type", "Record-level exceptions discovered by validation rules.")
    if not issues.empty:
        figure = px.bar(
            issues.sort_values("affected_records"),
            x="affected_records",
            y="issue_type",
            orientation="h",
            color="severity_weight",
            color_continuous_scale=["#f4d58d", "#e05d44", "#8f2d25"],
        )
        st.plotly_chart(base_layout(figure, 390), width="stretch", config=PLOT_CONFIG)
    else:
        st.info("No validation issues detected.")
with right:
    section_header("Quality dimensions")
    dimensions = {
        "Completeness": 100 - 100 * (events["case_id"].isna() | events["timestamp"].isna()).mean(),
        "Validity": 100 - 100 * (events["duration_minutes"] < 0).mean(),
        "Uniqueness": 100 - 100 * events["event_id"].duplicated().mean(),
        "Referential integrity": 100 - 100 * (~events["user_id"].isin(users["user_id"])).mean(),
        "Process integrity": score,
    }
    dimension_frame = (
        __import__("pandas")
        .DataFrame({"dimension": dimensions.keys(), "score": dimensions.values()})
        .sort_values("score")
    )
    figure = px.bar(
        dimension_frame,
        x="score",
        y="dimension",
        orientation="h",
        range_x=[0, 100],
        color="score",
        color_continuous_scale=["#e05d44", "#f4d58d", "#19a974"],
    )
    figure.update_coloraxes(showscale=False)
    st.plotly_chart(base_layout(figure, 390), width="stretch", config=PLOT_CONFIG)

section_header("Validation issue detail")
if not issues.empty:
    st.dataframe(
        issues,
        width="stretch",
        hide_index=True,
        column_config={
            "recommended_fix": st.column_config.TextColumn("Recommended fix", width="large"),
            "sample_records": st.column_config.TextColumn("Sample", width="medium"),
        },
    )
    st.download_button(
        "Download issue report",
        issues.to_csv(index=False),
        "worklens_data_quality_issues.csv",
        "text/csv",
    )
else:
    st.info("No exceptions to export.")
