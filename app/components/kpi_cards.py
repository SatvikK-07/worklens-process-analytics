from __future__ import annotations

import streamlit as st


def render_kpi(label: str, value: str, delta: str | None = None) -> None:
    """Render a standard metric card."""
    st.metric(label, value, delta)
