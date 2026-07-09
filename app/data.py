from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.analytics.automation_roi_engine import rank_automation_candidates
from src.analytics.bottleneck_engine import rank_bottlenecks
from src.analytics.rework_engine import detect_rework_loops
from src.ml.feature_engineering import build_case_features
from src.utils.config import settings


def missing_demo_artifacts() -> list[str]:
    required = [
        settings.data_dir / "cases.csv",
        settings.data_dir / "events.csv",
        settings.data_dir / "users.csv",
        settings.data_dir / "providers.csv",
        settings.data_dir / "predictions.csv",
    ]
    return [str(path) for path in required if not path.exists()]


def show_demo_setup_if_needed() -> None:
    missing = missing_demo_artifacts()
    if not missing:
        return
    st.warning(
        "Synthetic demo artifacts are not generated. This page is unavailable "
        "until the deterministic demo pipeline has run."
    )
    st.code("make data\nmake db\nmake train", language="bash")
    with st.expander("Missing files"):
        st.code("\n".join(missing))
    st.stop()


@st.cache_data(show_spinner=False)
def load_cases() -> pd.DataFrame:
    show_demo_setup_if_needed()
    frame = pd.read_csv(
        settings.data_dir / "cases.csv",
        parse_dates=["created_at", "closed_at"],
    )
    predictions_path = settings.data_dir / "predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        frame = frame.merge(predictions, on="case_id", how="left")
    return frame


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    show_demo_setup_if_needed()
    return pd.read_csv(settings.data_dir / "events.csv", parse_dates=["timestamp"])


@st.cache_data(show_spinner=False)
def load_users() -> pd.DataFrame:
    show_demo_setup_if_needed()
    return pd.read_csv(settings.data_dir / "users.csv")


@st.cache_data(show_spinner=False)
def load_providers() -> pd.DataFrame:
    show_demo_setup_if_needed()
    return pd.read_csv(settings.data_dir / "providers.csv")


@st.cache_data(show_spinner="Calculating activity performance…")
def load_bottlenecks() -> pd.DataFrame:
    return rank_bottlenecks(load_events(), load_cases())


@st.cache_data(show_spinner="Detecting rework loops…")
def load_rework_loops() -> pd.DataFrame:
    return detect_rework_loops(load_events())


@st.cache_data(show_spinner=False)
def load_automation_candidates() -> pd.DataFrame:
    return rank_automation_candidates(load_events(), load_cases())


@st.cache_data(show_spinner="Building model features…")
def load_features() -> pd.DataFrame:
    return build_case_features(load_cases(), load_events(), load_providers())


@st.cache_resource(show_spinner=False)
def load_model_artifact(name: str) -> dict:
    path = settings.model_dir / f"{name}.pkl"
    if not path.exists():
        st.warning(f"Model artifact `{name}` is missing. Run `make train`.")
        st.stop()
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_model_metrics(name: str) -> dict:
    path = settings.model_dir / f"{name}_metrics.json"
    if not path.exists():
        st.warning(f"Model metrics for `{name}` are missing. Run `make train`.")
        st.stop()
    return json.loads(path.read_text())
