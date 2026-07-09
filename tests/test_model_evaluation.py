from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import src.ml.train_completion_model as completion_module
import src.ml.train_sla_model as sla_module
from src.ml.anomaly_detection import anomaly_reason, train_anomaly_model
from src.ml.explainability import (
    global_feature_importance,
    local_shap_explanation,
)
from src.ml.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.ml.train_completion_model import regression_metrics
from src.ml.train_sla_model import (
    classification_metrics,
    select_operating_threshold,
    threshold_analysis,
)


def modeling_frame(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "case_id": [f"C-{index:04d}" for index in range(rows)],
            "created_at": pd.date_range("2025-01-01", periods=rows, freq="h"),
        }
    )
    for column in CATEGORICAL_FEATURES:
        frame[column] = np.where(np.arange(rows) % 2, f"{column}-A", f"{column}-B")
    for index, column in enumerate(NUMERIC_FEATURES):
        frame[column] = rng.normal(index + 2, 0.5, rows).clip(0)
    signal = frame["queue_wait_time_total"] + frame["historical_provider_delay_rate"]
    frame["sla_breached"] = (signal > signal.quantile(0.70)).astype(int)
    frame["total_duration_hours"] = (
        12 + frame["queue_wait_time_total"] * 2 + rng.normal(0, 2, rows)
    ).clip(1)
    frame["anomaly_label"] = 0
    return frame


def test_classification_metric_output_shape() -> None:
    target = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.3, 0.7, 0.9])
    metrics = classification_metrics(target, probability, 0.5)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["roc_auc"] == 1.0


def test_threshold_analysis_and_selection() -> None:
    target = pd.Series([0, 0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.4, 0.55, 0.75, 0.9])
    rows = threshold_analysis(target, probability)
    selected = select_operating_threshold(rows)
    assert len(rows) == 17
    assert 0.1 <= selected <= 0.9


def test_regression_metrics_are_zero_for_exact_prediction() -> None:
    target = pd.Series([1.0, 2.0, 3.0])
    metrics = regression_metrics(target, target.to_numpy())
    assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": 1.0, "mape": 0.0}


def test_temporal_sla_training_produces_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sla_module,
        "_candidate_estimators",
        lambda seed: {
            "Majority Baseline": DummyClassifier(strategy="prior"),
            "Logistic Regression": LogisticRegression(max_iter=300),
        },
    )
    monkeypatch.setattr(sla_module, "save_model", lambda *args, **kwargs: None)
    artifact, probability, metadata = sla_module.train_sla_models(modeling_frame())
    assert len(probability) == 160
    assert artifact["feature_snapshot"].startswith("first_5")
    assert metadata["validation_strategy"].startswith("Chronological")
    assert "random" in metadata["split_comparison"]
    sample = modeling_frame(40)
    global_result = global_feature_importance(artifact, 5, sample=sample)
    local_result = local_shap_explanation(artifact, sample.iloc[0], background=sample, top_n=5)
    assert len(global_result) == 5
    assert len(local_result) == 5


def test_temporal_completion_training_produces_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        completion_module,
        "_candidate_estimators",
        lambda seed: {
            "Median Baseline": DummyRegressor(strategy="median"),
            "Linear Regression": LinearRegression(),
        },
    )
    monkeypatch.setattr(completion_module, "save_model", lambda *args, **kwargs: None)
    artifact, prediction, metadata = completion_module.train_completion_models(modeling_frame())
    assert len(prediction) == 160
    assert artifact["feature_snapshot"].startswith("first_5")
    assert metadata["models"]["Median Baseline"]["mae"] > 0


def test_anomaly_training_returns_bounded_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ml.anomaly_detection as anomaly_module

    monkeypatch.setattr(anomaly_module, "save_model", lambda *args, **kwargs: None)
    _, scores, metadata = train_anomaly_model(modeling_frame())
    assert scores.between(0, 1).all()
    assert metadata["flagged_cases"] > 0


def test_anomaly_reason_lists_operational_signals() -> None:
    row = pd.Series(
        {
            "handoff_count": 12,
            "rework_count": 4,
            "provider_clarification_count": 3,
            "queue_wait_time_total": 90,
        }
    )
    reason = anomaly_reason(row)
    assert "12 handoffs" in reason
    assert "4 rework loops" in reason
