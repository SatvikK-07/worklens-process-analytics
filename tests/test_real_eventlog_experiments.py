from __future__ import annotations

import pandas as pd
import pytest

from real_eventlog_experiments.src.evaluate_classification import (
    classification_metrics,
)
from real_eventlog_experiments.src.evaluate_regression import regression_metrics
from real_eventlog_experiments.src.load_eventlog import normalize_event_log
from real_eventlog_experiments.src.prefix_features import (
    FORBIDDEN_PREFIX_FEATURES,
    assert_prefix_safe,
    build_prefix_features,
    fit_historical_prefix_encoders,
    prefix_feature_columns,
    transform_with_historical_prefix_encoders,
)
from real_eventlog_experiments.src.temporal_split import temporal_split
from real_eventlog_experiments.src.train_long_case_classifier import (
    run_long_case_experiment,
)
from real_eventlog_experiments.src.train_remaining_time_regressor import (
    run_remaining_time_experiment,
)


def event_frame() -> pd.DataFrame:
    rows = []
    for case_number in range(10):
        for event_number, activity in enumerate(["Start", "Review", "End"]):
            rows.append(
                {
                    "event_id": f"E-{case_number}-{event_number}",
                    "case_id": f"C-{case_number}",
                    "activity": activity,
                    "timestamp": (
                        pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(days=case_number, hours=event_number)
                    ),
                    "resource": "R-1",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "missing required columns"),
        (
            pd.DataFrame({"case_id": [None], "activity": ["A"], "timestamp": ["2025-01-01"]}),
            "case_id cannot be null",
        ),
        (
            pd.DataFrame({"case_id": ["C"], "activity": [None], "timestamp": ["2025-01-01"]}),
            "activity cannot be null",
        ),
        (
            pd.DataFrame({"case_id": ["C"], "activity": ["A"], "timestamp": ["not-a-date"]}),
            "unparseable",
        ),
    ],
)
def test_event_log_validation_errors(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_event_log(frame)


def test_simultaneous_unknown_activities_are_supported() -> None:
    frame = event_frame().iloc[:3].copy()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    frame.loc[1, "activity"] = "Previously unseen activity"
    normalized = normalize_event_log(frame)
    assert len(normalized) == 3
    assert "Previously unseen activity" in set(normalized["activity"])


def test_short_cases_are_excluded_from_prefix_features() -> None:
    frame = event_frame()
    short = frame.iloc[[0]].assign(case_id="SHORT", event_id="SHORT-1")
    features = build_prefix_features(pd.concat([frame, short]), prefix_length=3)
    assert "SHORT" not in set(features["case_id"])
    assert len(features) == 10


def test_rich_prefix_features_are_present_and_targets_are_not_model_inputs() -> None:
    features = build_prefix_features(event_frame(), prefix_length=3)
    expected = {
        "prefix_len",
        "elapsed_hours",
        "first_activity",
        "prefix_path",
        "gap_1_2_hours",
        "gap_2_3_hours",
        "mean_gap_hours",
        "min_gap_hours",
        "max_gap_hours",
        "std_gap_hours",
        "start_month",
        "weekend_start",
        "observed_rework_count_so_far",
        "observed_transition_count",
    }
    assert expected.issubset(features.columns)
    categorical, numeric = prefix_feature_columns(3)
    assert not FORBIDDEN_PREFIX_FEATURES.intersection(categorical + numeric)


def test_missing_optional_resource_uses_unknown() -> None:
    features = build_prefix_features(event_frame().drop(columns="resource"), prefix_length=3)
    assert set(features["last_resource"]) == {"Unknown"}


def test_prefix_feature_guard_rejects_targets() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_prefix_safe(["activity_1", "remaining_time_hours"])


def test_historical_encodings_ignore_validation_targets_and_default_unseen() -> None:
    features = build_prefix_features(modeling_event_frame(40), prefix_length=3)
    threshold = features.iloc[:28]["total_case_duration_hours"].median()
    features["long_case"] = (features["total_case_duration_hours"] > threshold).astype(int)
    train = features.iloc[:28].copy()
    validation = features.iloc[28:].copy()
    encoders = fit_historical_prefix_encoders(
        train,
        train,
        target_column="long_case",
        smoothing=10,
    )
    original = transform_with_historical_prefix_encoders(validation, encoders)
    changed_targets = validation.copy()
    changed_targets["long_case"] = 1 - changed_targets["long_case"]
    changed_targets["total_case_duration_hours"] *= 100
    changed = transform_with_historical_prefix_encoders(changed_targets, encoders)
    columns = [
        "last_activity_train_median_duration",
        "last_activity_train_long_case_rate",
        "prefix_path_train_long_case_rate_smoothed",
        "activity_set_train_median_duration",
        "resource_train_median_duration",
        "resource_train_long_case_rate",
    ]
    pd.testing.assert_frame_equal(original[columns], changed[columns])

    unseen = validation.iloc[[0]].copy()
    unseen["last_activity"] = "Never Seen"
    unseen["prefix_path"] = "Never → Seen"
    unseen["activity_set"] = "Never | Seen"
    unseen["last_resource"] = "Never Seen Resource"
    transformed = transform_with_historical_prefix_encoders(unseen, encoders).iloc[0]
    assert transformed["last_activity_train_median_duration"] == encoders.global_duration_median
    assert transformed["prefix_path_train_long_case_rate_smoothed"] == encoders.global_target_rate


def test_leave_one_out_encoding_does_not_copy_unique_row_target() -> None:
    features = build_prefix_features(modeling_event_frame(12), prefix_length=3)
    features["long_case"] = [0, 1] * 6
    features["prefix_path"] = [f"unique-{index}" for index in range(len(features))]
    encoders = fit_historical_prefix_encoders(
        features,
        features,
        target_column="long_case",
    )
    transformed = transform_with_historical_prefix_encoders(
        features,
        encoders,
        leave_one_out=True,
    )
    assert transformed["prefix_path_train_long_case_rate_smoothed"].nunique() == 1
    assert (
        transformed["prefix_path_train_long_case_rate_smoothed"].iloc[0]
        == encoders.global_target_rate
    )


def test_temporal_split_preserves_order() -> None:
    features = build_prefix_features(event_frame(), prefix_length=3)
    splits = temporal_split(features)
    assert splits.train["case_start"].max() <= splits.validation["case_start"].min()
    assert splits.validation["case_start"].max() <= splits.test["case_start"].min()


def test_baseline_metric_helpers() -> None:
    classification = classification_metrics(
        pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.2, 0.8, 0.9]).to_numpy(), 0.5
    )
    regression = regression_metrics(
        pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]).to_numpy()
    )
    assert classification["roc_auc"] == 1.0
    assert classification["confusion_matrix"] == [[2, 0], [0, 2]]
    assert regression["mae"] == 0.0
    assert regression["median_absolute_error"] == 0.0


def modeling_event_frame(case_count: int = 120) -> pd.DataFrame:
    rows = []
    for case_number in range(case_count):
        start = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(days=case_number)
        duration = [12, 48, 120, 240][case_number % 4]
        activities = ["Start", f"Route {case_number % 3}", "Review", "Decision", "End"]
        offsets = [0, 1, 3, duration - 1, duration]
        for event_number, (activity, offset) in enumerate(zip(activities, offsets, strict=True)):
            rows.append(
                {
                    "event_id": f"M-{case_number}-{event_number}",
                    "case_id": f"M-{case_number:03d}",
                    "activity": activity,
                    "timestamp": start + pd.Timedelta(hours=offset),
                    "resource": f"R-{case_number % 5}",
                    "lifecycle": "complete",
                }
            )
    return pd.DataFrame(rows)


def test_real_experiment_training_paths_return_required_metrics() -> None:
    events = modeling_event_frame()
    classification = run_long_case_experiment(events, prefix_length=3, seed=17)
    regression = run_remaining_time_experiment(events, prefix_length=3, seed=17)
    selected_classification = classification["models"][classification["selected_model"]]
    selected_regression = regression["models"][regression["selected_model"]]
    assert {"roc_auc", "pr_auc", "brier_score", "confusion_matrix"}.issubset(
        selected_classification
    )
    assert {"mae", "rmse", "median_absolute_error", "r2"}.issubset(selected_regression)
    assert classification["validation_strategy"] == "temporal_70_15_15"
    assert regression["validation_strategy"] == "temporal_70_15_15"
