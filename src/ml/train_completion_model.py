from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
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
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


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


def regression_metrics(target: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    denominator = np.maximum(np.abs(target.to_numpy()), 1)
    return {
        "mae": round(float(mean_absolute_error(target, prediction)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(target, prediction))), 4),
        "r2": round(float(r2_score(target, prediction)), 4),
        "mape": round(float(np.mean(np.abs(target.to_numpy() - prediction) / denominator)), 4),
    }


def _candidate_estimators(random_seed: int) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Median Baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=260,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.84,
            colsample_bytree=0.84,
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=-1,
        )
    return candidates


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor()), ("model", estimator)])


def train_completion_models(
    features: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.Series, dict[str, Any]]:
    validate_features(MODEL_FEATURES, "remaining_time_prediction")
    temporal = temporal_train_validation_test_split(features)
    candidates = _candidate_estimators(random_seed)
    fitted: dict[str, Pipeline] = {}
    validation_evaluations: dict[str, dict[str, float]] = {}
    test_evaluations: dict[str, dict[str, float]] = {}

    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator)
        pipeline.fit(temporal.train[MODEL_FEATURES], temporal.train["total_duration_hours"])
        validation_prediction = np.clip(
            pipeline.predict(temporal.validation[MODEL_FEATURES]), 0, None
        )
        test_prediction = np.clip(pipeline.predict(temporal.test[MODEL_FEATURES]), 0, None)
        validation_evaluations[name] = regression_metrics(
            temporal.validation["total_duration_hours"], validation_prediction
        )
        test_evaluations[name] = regression_metrics(
            temporal.test["total_duration_hours"], test_prediction
        )
        fitted[name] = pipeline

    selectable = [name for name in candidates if name != "Median Baseline"]
    selected_name = min(selectable, key=lambda name: validation_evaluations[name]["mae"])
    train_and_validation = pd.concat([temporal.train, temporal.validation])
    selected = clone(_pipeline(candidates[selected_name]))
    selected.fit(
        train_and_validation[MODEL_FEATURES],
        train_and_validation["total_duration_hours"],
    )
    temporal_prediction = np.clip(selected.predict(temporal.test[MODEL_FEATURES]), 0, None)
    test_evaluations[selected_name] = regression_metrics(
        temporal.test["total_duration_hours"], temporal_prediction
    )

    random_splits = random_train_validation_test_split(
        features, target=None, random_seed=random_seed
    )
    random_train = pd.concat([random_splits.train, random_splits.validation])
    random_model = clone(_pipeline(candidates[selected_name]))
    random_model.fit(random_train[MODEL_FEATURES], random_train["total_duration_hours"])
    random_prediction = np.clip(random_model.predict(random_splits.test[MODEL_FEATURES]), 0, None)
    random_metrics = regression_metrics(
        random_splits.test["total_duration_hours"], random_prediction
    )

    all_prediction = pd.Series(
        np.clip(selected.predict(features[MODEL_FEATURES]), 0, None),
        index=features.index,
        name="predicted_completion_hours",
    )
    metadata = {
        "selected_model": selected_name,
        "selection_policy": "Lowest validation MAE on the temporal split",
        "validation_strategy": "Chronological 70/15/15 train/validation/test",
        "split_boundaries": temporal_boundaries(temporal),
        "test_rows": len(temporal.test),
        "selected_test_median_absolute_error": round(
            float(
                median_absolute_error(temporal.test["total_duration_hours"], temporal_prediction)
            ),
            4,
        ),
        "models": test_evaluations,
        "validation_models": validation_evaluations,
        "split_comparison": {
            "temporal": test_evaluations[selected_name],
            "random": random_metrics,
        },
    }
    artifact = {
        "pipeline": selected,
        "features": MODEL_FEATURES,
        "feature_snapshot": "first_5_events_plus_strictly_prior_history",
    }
    save_model("completion_time_model", artifact, metadata)
    return artifact, all_prediction, metadata
