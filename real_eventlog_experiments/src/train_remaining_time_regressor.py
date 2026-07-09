from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from real_eventlog_experiments.src.baselines import regression_models
from real_eventlog_experiments.src.evaluate_regression import (
    error_by_duration_bucket,
    regression_metrics,
    residual_summary,
)
from real_eventlog_experiments.src.prefix_features import (
    assert_prefix_safe,
    build_prefix_features,
    fit_historical_prefix_encoders,
    prefix_feature_columns,
    transform_with_historical_prefix_encoders,
)
from real_eventlog_experiments.src.temporal_split import random_split, temporal_split


def _pipeline(model: Any, categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _add_encoding_target(features: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    splits = temporal_split(features)
    threshold = float(splits.train["total_case_duration_hours"].quantile(0.75))
    enriched = features.copy()
    enriched["long_case_q75"] = (enriched["total_case_duration_hours"] > threshold).astype(int)
    return enriched, threshold


def _encode_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encoders = fit_historical_prefix_encoders(
        train,
        train,
        target_column="long_case_q75",
    )
    return (
        transform_with_historical_prefix_encoders(train, encoders, leave_one_out=True),
        transform_with_historical_prefix_encoders(validation, encoders),
        transform_with_historical_prefix_encoders(test, encoders),
    )


def run_remaining_time_experiment(
    events: pd.DataFrame,
    prefix_length: int = 3,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    base_features = build_prefix_features(events, prefix_length)
    features, long_threshold = _add_encoding_target(base_features)
    categorical, numeric = prefix_feature_columns(prefix_length)
    model_features = categorical + numeric
    assert_prefix_safe(model_features)
    splits = temporal_split(features)
    encoded_train, encoded_validation, _ = _encode_splits(
        splits.train,
        splits.validation,
        splits.test,
    )

    candidates = regression_models(seed)
    validation_metrics: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator, categorical, numeric)
        pipeline.fit(
            encoded_train[model_features],
            encoded_train["remaining_time_hours"],
        )
        prediction = np.clip(pipeline.predict(encoded_validation[model_features]), 0, None)
        validation_metrics[name] = regression_metrics(
            encoded_validation["remaining_time_hours"], prediction
        )
    selected_name = min(
        candidates,
        key=lambda name: validation_metrics[name]["mae"],
    )

    train_validation = pd.concat([splits.train, splits.validation])
    final_encoders = fit_historical_prefix_encoders(
        train_validation,
        train_validation,
        target_column="long_case_q75",
    )
    encoded_train_validation = transform_with_historical_prefix_encoders(
        train_validation,
        final_encoders,
        leave_one_out=True,
    )
    encoded_test = transform_with_historical_prefix_encoders(splits.test, final_encoders)
    test_metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    for name, estimator in candidates.items():
        pipeline = clone(_pipeline(estimator, categorical, numeric))
        pipeline.fit(
            encoded_train_validation[model_features],
            encoded_train_validation["remaining_time_hours"],
        )
        prediction = np.clip(pipeline.predict(encoded_test[model_features]), 0, None)
        predictions_by_model[name] = prediction
        test_metrics[name] = regression_metrics(encoded_test["remaining_time_hours"], prediction)
        fitted[name] = pipeline
    baseline_mae = test_metrics["Median Baseline"]["mae"]
    test_prediction = predictions_by_model[selected_name]
    selected_test_metrics = regression_metrics(
        encoded_test["remaining_time_hours"],
        test_prediction,
        baseline_mae=baseline_mae,
    )
    test_metrics[selected_name] = selected_test_metrics

    random_splits = random_split(features, target=None, seed=seed)
    random_train_validation = pd.concat([random_splits.train, random_splits.validation])
    random_encoders = fit_historical_prefix_encoders(
        random_train_validation,
        random_train_validation,
        target_column="long_case_q75",
    )
    encoded_random_train = transform_with_historical_prefix_encoders(
        random_train_validation,
        random_encoders,
        leave_one_out=True,
    )
    encoded_random_test = transform_with_historical_prefix_encoders(
        random_splits.test,
        random_encoders,
    )
    random_model = _pipeline(regression_models(seed)[selected_name], categorical, numeric)
    random_model.fit(
        encoded_random_train[model_features],
        encoded_random_train["remaining_time_hours"],
    )
    random_prediction = np.clip(random_model.predict(encoded_random_test[model_features]), 0, None)
    random_metrics = regression_metrics(
        encoded_random_test["remaining_time_hours"], random_prediction
    )

    predictions = encoded_test[
        ["case_id", "case_start", "remaining_time_hours", "process_variant"]
    ].copy()
    predictions["prediction"] = test_prediction
    predictions["absolute_error"] = (
        predictions["remaining_time_hours"] - predictions["prediction"]
    ).abs()
    variant_errors = (
        predictions.groupby("process_variant", as_index=False)
        .agg(cases=("case_id", "count"), mae=("absolute_error", "mean"))
        .query("cases >= 2")
        .sort_values("mae", ascending=False)
        .head(20)
        .to_dict("records")
    )
    result = {
        "task": "remaining_time_regression",
        "target": "remaining_time_hours",
        "target_transform": (
            "Non-baseline estimators train on log1p remaining hours and return expm1 predictions"
        ),
        "prefix_length": prefix_length,
        "eligible_cases": len(features),
        "selected_model": selected_name,
        "selection_policy": "Lowest validation MAE on chronological validation window",
        "validation_strategy": "temporal_70_15_15",
        "split_rows": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "features": model_features,
        "historical_encoding_long_case_threshold_hours": long_threshold,
        "historical_encoding_policy": (
            "Fit on train only for selection; leave-one-out on fitting rows; "
            "refit on train+validation only before final test"
        ),
        "models": test_metrics,
        "validation_models": validation_metrics,
        "temporal_vs_random": {
            "temporal": selected_test_metrics,
            "random": random_metrics,
        },
        "beats_median_baseline": bool(
            selected_test_metrics["mae"] < test_metrics["Median Baseline"]["mae"]
        ),
        "baseline_mae": baseline_mae,
        "error_by_duration_bucket": error_by_duration_bucket(
            encoded_test["remaining_time_hours"], test_prediction
        ),
        "residual_summary": residual_summary(encoded_test["remaining_time_hours"], test_prediction),
        "error_by_process_variant": variant_errors,
        "top_error_cases": predictions.nlargest(10, "absolute_error")[
            ["case_id", "remaining_time_hours", "prediction", "absolute_error"]
        ].to_dict("records"),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"remaining_time_n{prefix_length}"
        (output_dir / f"{stem}_metrics.json").write_text(json.dumps(result, indent=2, default=str))
        predictions.to_csv(output_dir / f"{stem}_predictions.csv", index=False)
    return result
