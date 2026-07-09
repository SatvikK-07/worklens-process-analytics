import streamlit as st

from app.components.page import page_header
from app.components.styles import apply_enterprise_theme
from src.ml.feature_registry import feature_registry_frame

apply_enterprise_theme()
page_header(
    "Feature governance",
    "Leakage audit",
    "Inspect when every feature becomes available and which modelling tasks may use it.",
    track="Governance across synthetic and real validation tracks",
)

audit = feature_registry_frame()
unsafe = audit[audit["uses_future_information"]]
metrics = st.columns(3)
metrics[0].metric("Registered features", len(audit))
metrics[1].metric("Prefix-safe features", (~audit["uses_future_information"]).sum())
metrics[2].metric("Post-completion fields", len(unsafe))
st.dataframe(audit, width="stretch", hide_index=True)
st.info(
    "Training fails when an unregistered or task-unsafe feature is supplied. "
    "Targets and final-case fields remain available only for evaluation or retrospective analysis."
)
