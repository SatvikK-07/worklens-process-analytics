from __future__ import annotations

import streamlit as st


def render_insight(title: str, message: str) -> None:
    st.info(f"**{title}**  \n{message}")
