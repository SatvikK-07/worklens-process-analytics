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

from real_eventlog_experiments.src.baselines import classification_models
from real_eventlog_experiments.src.calibration import calibration_data
from real_eventlog_experiments.src.evaluate_classification import (
    choose_threshold,
    classification_metrics,
    lift_table,
    threshold_table,
)
from real_eventlog_experiments.src.explainability import (
    local_associated_drivers,
    transformed_feature_importance,
)
from real_eventlog_experiments.src.prefix_features import (
    assert_prefix_safe,
    build_prefix_features,
    fit_historical_prefix_encoders,
    prefix_feature_columns,
    transform_with_historical_prefix_encoders,
)
from real_eventlog_experiments.src.temporal_split import random_split, temporal_split

TARGET_DESCRIPTIONS = {
    "long_case_q75": "Duration exceeds the training-window 75th percentile",
    "long_case_median": "Duration exceeds the training-window median",
    "open_after_7_days": "Case remains open 7 days after case start",
    "open_after_14_days": "Case remains open 14 days after case start",
    "open_after_30_days": "Case remains open 30 days after case start",
}


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


def _add_target_variants(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    splits = temporal_split(features)
    train_duration = splits.train["total_case_duration_hours"]
    thresholds = {
        "long_case_q75": float(train_duration.quantile(0.75)),
        "long_case_median": float(train_duration.median()),
        "open_after_7_days": 7 * 24.0,
        "open_after_14_days": 14 * 24.0,
        "open_after_30_days": 30 * 24.0,
    }
    enriched = features.copy()
    for target, threshold in thresholds.items():
        enriched[target] = (enriched["total_case_duration_hours"] > threshold).astype(int)
    return enriched, thresholds


def _validation_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["pr_auc_lift_over_prevalence"]),
        (float(metrics["balanced_accuracy"]) if metrics["balanced_accuracy"] is not None else -1.0),
        float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else -1.0,
    )


def _fit_validation_variant(
    features: pd.DataFrame,
    target: str,
    categorical: list[str],
    numeric: list[str],
    seed: int,
) -> dict[str, Any] | None:
    splits = temporal_split(features)
    if splits.train[target].nunique() < 2:
        return None
    encoders = fit_historical_prefix_encoders(
        splits.train,
        splits.train,
        target_column=target,
    )
    train = transform_with_historical_prefix_encoders(splits.train, encoders, leave_one_out=True)
    validation = transform_with_historical_prefix_encoders(splits.validation, encoders)
    candidates = classification_models(seed)
    fitted: dict[str, Pipeline] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}
    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator, categorical, numeric)
        pipeline.fit(train[categorical + numeric], train[target])
        probability = pipeline.predict_proba(validation[categorical + numeric])[:, 1]
        validation_metrics[name] = classification_metrics(validation[target], probability, 0.5)
        fitted[name] = pipeline
    selectable = [name for name in candidates if name != "Majority Baseline"]
    selected_name = max(
        selectable,
        key=lambda name: _validation_score(validation_metrics[name]),
    )
    validation_probability = fitted[selected_name].predict_proba(validation[categorical + numeric])[
        :, 1
    ]
    threshold_rows = threshold_table(validation[target], validation_probability)
    return {
        "selected_model": selected_name,
        "validation_models": validation_metrics,
        "threshold_table": threshold_rows,
        "selected_threshold": choose_threshold(threshold_rows),
        "selection_score": _validation_score(validation_metrics[selected_name]),
    }


def _evaluate_variant(
    features: pd.DataFrame,
    target: str,
    validation_state: dict[str, Any],
    categorical: list[str],
    numeric: list[str],
    seed: int,
) -> dict[str, Any]:
    splits = temporal_split(features)
    train_validation = pd.concat([splits.train, splits.validation])
    encoders = fit_historical_prefix_encoders(
        train_validation,
        train_validation,
        target_column=target,
    )
    encoded_train_validation = transform_with_historical_prefix_encoders(
        train_validation,
        encoders,
        leave_one_out=True,
    )
    encoded_test = transform_with_historical_prefix_encoders(splits.test, encoders)
    candidates = classification_models(seed)
    test_metrics: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    threshold = float(validation_state["selected_threshold"])
    model_features = categorical + numeric
    for name, estimator in candidates.items():
        pipeline = clone(_pipeline(estimator, categorical, numeric))
        pipeline.fit(encoded_train_validation[model_features], train_validation[target])
        probability = pipeline.predict_proba(encoded_test[model_features])[:, 1]
        test_metrics[name] = classification_metrics(
            encoded_test[target],
            probability,
            threshold if name == validation_state["selected_model"] else 0.5,
        )
        fitted[name] = pipeline
    selected = fitted[validation_state["selected_model"]]
    probability = selected.predict_proba(encoded_test[model_features])[:, 1]
    return {
        "models": test_metrics,
        "selected_pipeline": selected,
        "test_probability": probability,
        "encoded_test": encoded_test,
        "encoded_train_validation": encoded_train_validation,
        "calibration": calibration_data(encoded_test[target], probability),
        "lift_table": lift_table(encoded_test[target], probability),
    }


def _random_comparison(
    features: pd.DataFrame,
    target: str,
    selected_name: str,
    threshold: float,
    categorical: list[str],
    numeric: list[str],
    seed: int,
) -> dict[str, Any]:
    splits = random_split(features, target, seed)
    train_validation = pd.concat([splits.train, splits.validation])
    encoders = fit_historical_prefix_encoders(
        train_validation,
        train_validation,
        target_column=target,
    )
    encoded_train = transform_with_historical_prefix_encoders(
        train_validation, encoders, leave_one_out=True
    )
    encoded_test = transform_with_historical_prefix_encoders(splits.test, encoders)
    estimator = classification_models(seed)[selected_name]
    pipeline = _pipeline(estimator, categorical, numeric)
    model_features = categorical + numeric
    pipeline.fit(encoded_train[model_features], encoded_train[target])
    probability = pipeline.predict_proba(encoded_test[model_features])[:, 1]
    return classification_metrics(encoded_test[target], probability, threshold)


def run_long_case_experiment(
    events: pd.DataFrame,
    prefix_length: int = 3,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    base_features = build_prefix_features(events, prefix_length)
    features, thresholds = _add_target_variants(base_features)
    categorical, numeric = prefix_feature_columns(prefix_length)
    model_features = categorical + numeric
    assert_prefix_safe(model_features)

    validation_states: dict[str, dict[str, Any]] = {}
    for target in TARGET_DESCRIPTIONS:
        state = _fit_validation_variant(features, target, categorical, numeric, seed)
        if state is not None:
            validation_states[target] = state
    if not validation_states:
        raise ValueError("No target variant has both classes in the training window")
    selected_target = max(
        validation_states,
        key=lambda target: validation_states[target]["selection_score"],
    )

    evaluations = {
        target: _evaluate_variant(
            features,
            target,
            state,
            categorical,
            numeric,
            seed,
        )
        for target, state in validation_states.items()
    }
    primary_state = validation_states[selected_target]
    primary = evaluations[selected_target]
    selected_name = primary_state["selected_model"]
    threshold = float(primary_state["selected_threshold"])
    selected_test_metrics = primary["models"][selected_name]
    random_metrics = _random_comparison(
        features,
        selected_target,
        selected_name,
        threshold,
        categorical,
        numeric,
        seed,
    )

    test = primary["encoded_test"]
    probability = primary["test_probability"]
    predictions = test[
        [
            "case_id",
            "case_start",
            "total_case_duration_hours",
            selected_target,
        ]
    ].copy()
    predictions = predictions.rename(columns={selected_target: "target"})
    predictions["target_variant"] = selected_target
    predictions["probability"] = probability
    predictions["prediction"] = (probability >= threshold).astype(int)

    selected_pipeline = primary["selected_pipeline"]
    importance = transformed_feature_importance(selected_pipeline)
    representative_index = int(np.argmax(probability))
    representative = test.iloc[[representative_index]]
    local_drivers = local_associated_drivers(
        selected_pipeline,
        representative,
        primary["encoded_train_validation"],
        model_features,
    )
    target_variant_results = {}
    for target, state in validation_states.items():
        evaluation = evaluations[target]
        selected_model = state["selected_model"]
        target_variant_results[target] = {
            "description": TARGET_DESCRIPTIONS[target],
            "threshold_hours": thresholds[target],
            "selected_model": selected_model,
            "selected_threshold": state["selected_threshold"],
            "validation_metrics": state["validation_models"][selected_model],
            "test_metrics": evaluation["models"][selected_model],
        }
    result = {
        "task": "early_case_classification",
        "prefix_length": prefix_length,
        "eligible_cases": len(features),
        "selected_target": selected_target,
        "target_definition": TARGET_DESCRIPTIONS[selected_target],
        "target_threshold_hours": thresholds[selected_target],
        "long_case_threshold_hours": thresholds["long_case_q75"],
        "target_variants": target_variant_results,
        "selected_model": selected_name,
        "selected_threshold": threshold,
        "model_selection": (
            "Target and estimator selected on validation PR-AUC lift, balanced "
            "accuracy, then ROC-AUC; test metrics were not used for selection"
        ),
        "validation_strategy": "temporal_70_15_15",
        "split_rows": {
            "train": len(temporal_split(features).train),
            "validation": len(temporal_split(features).validation),
            "test": len(temporal_split(features).test),
        },
        "features": model_features,
        "historical_encoding_policy": (
            "Fit on train only for selection; leave-one-out on fitting rows; "
            "refit on train+validation only before final test"
        ),
        "models": primary["models"],
        "validation_models": primary_state["validation_models"],
        "temporal_vs_random": {
            "temporal": selected_test_metrics,
            "random": random_metrics,
        },
        "calibration": primary["calibration"],
        "threshold_table": primary_state["threshold_table"],
        "lift_table": primary["lift_table"],
        "representative_case": {
            "case_id": str(representative.iloc[0]["case_id"]),
            "probability": float(probability[representative_index]),
            "top_associated_drivers": local_drivers.to_dict("records"),
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"long_case_n{prefix_length}"
        (output_dir / f"{stem}_metrics.json").write_text(json.dumps(result, indent=2, default=str))
        predictions.to_csv(output_dir / f"{stem}_predictions.csv", index=False)
        importance.to_csv(output_dir / f"{stem}_feature_importance.csv", index=False)
    return result
