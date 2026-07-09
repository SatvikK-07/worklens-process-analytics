from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from src.ml.feature_registry import validate_features
from src.ml.model_registry import save_model
from src.ml.validation import (
    random_train_validation_test_split,
    temporal_boundaries,
    temporal_train_validation_test_split,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional runtime dependency
    XGBClassifier = None


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def classification_metrics(
    target: pd.Series, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(target, prediction)), 4),
        "precision": round(float(precision_score(target, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(target, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(target, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(target, probability)), 4),
        "pr_auc": round(float(average_precision_score(target, probability)), 4),
        "brier_score": round(float(brier_score_loss(target, probability)), 4),
        "confusion_matrix": confusion_matrix(target, prediction).tolist(),
        "decision_threshold": round(float(threshold), 3),
    }


def threshold_analysis(target: pd.Series, probability: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        prediction = probability >= threshold
        precision = precision_score(target, prediction, zero_division=0)
        recall = recall_score(target, prediction, zero_division=0)
        f2 = (5 * precision * recall / (4 * precision + recall)) if precision + recall else 0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f2": round(float(f2), 4),
            }
        )
    return rows


def select_operating_threshold(rows: list[dict[str, float]]) -> float:
    eligible = [row for row in rows if row["precision"] >= 0.30]
    selected = max(eligible or rows, key=lambda row: row["f2"])
    return float(selected["threshold"])


def _candidate_estimators(random_seed: int) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Majority Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=800, class_weight="balanced", random_state=random_seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=240,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.82,
            colsample_bytree=0.82,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
        )
    return candidates


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor()), ("model", estimator)])


def train_sla_models(
    features: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.Series, dict[str, Any]]:
    validate_features(MODEL_FEATURES, "sla_prediction")
    temporal = temporal_train_validation_test_split(features)
    candidates = _candidate_estimators(random_seed)
    validation_evaluations: dict[str, dict[str, Any]] = {}
    test_evaluations: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator)
        pipeline.fit(temporal.train[MODEL_FEATURES], temporal.train["sla_breached"])
        validation_probability = pipeline.predict_proba(temporal.validation[MODEL_FEATURES])[:, 1]
        test_probability = pipeline.predict_proba(temporal.test[MODEL_FEATURES])[:, 1]
        validation_evaluations[name] = classification_metrics(
            temporal.validation["sla_breached"], validation_probability, 0.35
        )
        test_evaluations[name] = classification_metrics(
            temporal.test["sla_breached"], test_probability, 0.35
        )
        fitted[name] = pipeline

    def business_score(name: str) -> float:
        values = validation_evaluations[name]
        return 0.40 * values["recall"] + 0.35 * values["roc_auc"] + 0.25 * values["pr_auc"]

    selectable = [name for name in candidates if name != "Majority Baseline"]
    selected_name = max(selectable, key=business_score)
    selected_validation_probability = fitted[selected_name].predict_proba(
        temporal.validation[MODEL_FEATURES]
    )[:, 1]
    threshold_rows = threshold_analysis(
        temporal.validation["sla_breached"], selected_validation_probability
    )
    selected_threshold = select_operating_threshold(threshold_rows)

    train_and_validation = pd.concat([temporal.train, temporal.validation])
    selected = clone(_pipeline(candidates[selected_name]))
    selected.fit(
        train_and_validation[MODEL_FEATURES],
        train_and_validation["sla_breached"],
    )
    temporal_test_probability = selected.predict_proba(temporal.test[MODEL_FEATURES])[:, 1]
    test_evaluations[selected_name] = classification_metrics(
        temporal.test["sla_breached"],
        temporal_test_probability,
        selected_threshold,
    )

    fraction_positive, mean_predicted = calibration_curve(
        temporal.test["sla_breached"],
        temporal_test_probability,
        n_bins=10,
        strategy="quantile",
    )

    random_splits = random_train_validation_test_split(
        features, target="sla_breached", random_seed=random_seed
    )
    random_train = pd.concat([random_splits.train, random_splits.validation])
    random_model = clone(_pipeline(candidates[selected_name]))
    random_model.fit(random_train[MODEL_FEATURES], random_train["sla_breached"])
    random_probability = random_model.predict_proba(random_splits.test[MODEL_FEATURES])[:, 1]
    random_metrics = classification_metrics(
        random_splits.test["sla_breached"], random_probability, selected_threshold
    )

    all_probability = pd.Series(
        selected.predict_proba(features[MODEL_FEATURES])[:, 1],
        index=features.index,
        name="sla_breach_probability",
    )
    metadata = {
        "selected_model": selected_name,
        "selection_policy": "Temporal validation: 40% recall + 35% ROC-AUC + 25% PR-AUC",
        "validation_strategy": "Chronological 70/15/15 train/validation/test",
        "split_boundaries": temporal_boundaries(temporal),
        "test_rows": len(temporal.test),
        "positive_rate": round(float(features["sla_breached"].mean()), 4),
        "models": test_evaluations,
        "validation_models": validation_evaluations,
        "split_comparison": {
            "temporal": test_evaluations[selected_name],
            "random": random_metrics,
        },
        "threshold_analysis": threshold_rows,
        "calibration_curve": {
            "mean_predicted_probability": mean_predicted.round(6).tolist(),
            "fraction_positive": fraction_positive.round(6).tolist(),
        },
    }
    artifact = {
        "pipeline": selected,
        "features": MODEL_FEATURES,
        "threshold": selected_threshold,
        "feature_snapshot": "first_5_events_plus_strictly_prior_history",
    }
    save_model("sla_breach_model", artifact, metadata)
    return artifact, all_probability, metadata
