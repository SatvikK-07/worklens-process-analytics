import streamlit as st

from app.components.page import page_header
from app.components.styles import apply_enterprise_theme

apply_enterprise_theme()
page_header(
    "Evidence guide",
    "Methodology and limitations",
    "Understand what the project demonstrates, validates, and deliberately does not claim.",
    track="Documentation — synthetic demo and real validation",
)

st.markdown(
    """
### Two evidence tracks

- **Synthetic healthcare claims:** deterministic product demonstration for process maps,
  bottlenecks, interventions, ROI, and API/dashboard workflows. Its model metrics are
  not evidence of real claims performance.
- **Public real event log:** independent modelling validation for first-N-event
  long-case classification and remaining-time regression.

### Leakage controls

Early models use case-start fields, the first five observed synthetic events, or the
first N real-log events, plus history computed strictly before case creation.
Completion fields and future events are rejected by the feature registry.

### Validation

Chronological train/validation/test splits are primary. Random splits are diagnostic
only. Model selection and threshold selection use validation data; the final test
window remains untouched until evaluation.

### Interpretation limits

Feature explanations are associations, not causal effects. Anomaly scores prioritize
review; they do not prove fraud, error, or misconduct. The application is a portfolio
project, not a production clinical or payment decision system.
"""
)
