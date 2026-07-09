import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.page import page_header, section_header
from app.components.styles import apply_enterprise_theme
from src.utils.config import settings

apply_enterprise_theme()
page_header(
    "Model governance",
    "Model monitoring and metrics",
    "Review saved evaluation evidence, split strategy, and artifact availability.",
)

rows = []
for name in [
    "sla_breach_model",
    "completion_time_model",
    "anomaly_model",
    "retrospective_anomaly_model",
]:
    path = settings.model_dir / f"{name}_metrics.json"
    if not path.exists():
        rows.append({"model": name, "status": "Missing — run make train"})
        continue
    metrics = json.loads(path.read_text())
    selected = metrics.get("selected_model")
    test = metrics.get("models", {}).get(selected, {}) if selected else {}
    rows.append(
        {
            "model": name,
            "status": "Available",
            "selected_estimator": selected or metrics.get("model"),
            "validation": metrics.get("validation_strategy", metrics.get("mode")),
            **test,
        }
    )

section_header("Synthetic model artifacts")
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.warning(
    "Artifact availability is not production monitoring. Drift, live outcomes, "
    "alerting, and retraining controls are documented as productionization work."
)

real_summary = (
    Path(__file__).resolve().parents[2]
    / "real_eventlog_experiments"
    / "results"
    / "experiment_summary.json"
)
section_header("Independent real-log evidence")
if real_summary.exists():
    st.json(json.loads(real_summary.read_text()), expanded=False)
else:
    st.info("No real event-log metrics artifact is available yet.")
