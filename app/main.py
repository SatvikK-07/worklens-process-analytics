from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.styles import apply_enterprise_theme
from app.data import missing_demo_artifacts

st.set_page_config(
    page_title="WorkLens AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_enterprise_theme()

with st.sidebar:
    st.markdown('<div class="brand-mark">W</div>', unsafe_allow_html=True)
    st.markdown("## WorkLens AI")
    st.caption("OPERATIONS INTELLIGENCE")
    st.divider()
    st.info("Use the page navigation above to move through the workspace.")
    st.caption("Synthetic product demo and independent real event-log validation")

st.markdown('<div class="eyebrow">WORKLENS AI / OVERVIEW</div>', unsafe_allow_html=True)
st.title("Workflow intelligence with explicit evidence boundaries.")
st.markdown(
    "The synthetic healthcare-claims track demonstrates the product. The public "
    "event-log track independently validates prefix-based modelling methods."
)

left, middle, right = st.columns(3)
with left:
    st.markdown(
        '<div class="feature-card"><span>01</span><h3>Discover</h3>'
        "<p>Map process variants and transition behavior from raw event logs.</p></div>",
        unsafe_allow_html=True,
    )
with middle:
    st.markdown(
        '<div class="feature-card"><span>02</span><h3>Prioritize</h3>'
        "<p>Rank bottlenecks, rework patterns, and automation opportunities.</p></div>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="feature-card"><span>03</span><h3>Intervene</h3>'
        "<p>Find high-risk cases early and give operations teams the next action.</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("### Platform readiness")
missing = missing_demo_artifacts()
if missing:
    st.warning("Synthetic demo artifacts are not currently generated.")
    st.code("make data\nmake db\nmake train\nmake run-streamlit", language="bash")
else:
    st.success("Synthetic demo artifacts are available.")
st.caption("Metrics shown in the application come from generated artifacts, not hard-coded claims.")

st.markdown(
    """
    <div class="insight-box">
      <small>START HERE</small>
      <p>Open <b>Executive Summary</b> for the synthetic product walkthrough.
      Open <b>Real Event-Log Validation</b> for independent modelling evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
