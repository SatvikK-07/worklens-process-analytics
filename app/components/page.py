from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def page_header(
    eyebrow: str,
    title: str,
    description: str,
    track: str = "Synthetic product demo",
) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow.upper()}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)
    st.info(f"Data track: {track}")


def section_header(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def empty_state(message: str) -> None:
    st.info(message)


def money(value: float, compact: bool = True) -> str:
    if compact and abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if compact and abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def hours(value: float) -> str:
    return f"{value:,.1f}h"


def base_layout(figure: go.Figure, height: int = 380) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=40, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#33443e"),
        hoverlabel=dict(bgcolor="#10231c", font_color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figure.update_xaxes(showgrid=False, linecolor="#dde5e0")
    figure.update_yaxes(gridcolor="#e8edea", zeroline=False)
    return figure
