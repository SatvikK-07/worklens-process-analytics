from __future__ import annotations

import pandas as pd
import pytest

from src.data_generation.generate_claims_event_log import GenerationConfig, generate_dataset
from src.ml.early_case_features import (
    CATEGORICAL_FEATURES,
    EARLY_NUMERIC_FEATURES,
    build_early_case_features,
)
from src.ml.feature_engineering import MODEL_FEATURES, build_case_features
from src.ml.leakage_audit import (
    FORBIDDEN_PREDICTORS,
    assert_no_forbidden_features,
    leakage_audit_table,
)
from src.ml.post_completion_features import build_post_completion_features
from src.ml.validation import (
    random_train_validation_test_split,
    temporal_boundaries,
    temporal_train_validation_test_split,
)


def test_model_feature_list_excludes_post_completion_fields() -> None:
    assert not FORBIDDEN_PREDICTORS.intersection(MODEL_FEATURES)


def test_leakage_guard_rejects_forbidden_predictor() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assert_no_forbidden_features([*MODEL_FEATURES, "total_duration_hours"])


def test_leakage_audit_marks_target_only() -> None:
    audit = leakage_audit_table().set_index("feature")
    assert audit.loc["sla_breached", "used_in_sla_model"] == "Target only"
    assert audit.loc["queue_wait_time_total", "available_at_prediction_time"] == "Yes"


def test_events_after_fifth_do_not_change_early_features() -> None:
    tables = generate_dataset(GenerationConfig(case_count=80, seed=31))
    baseline = build_early_case_features(
        tables["cases"], tables["events"], tables["providers"]
    ).set_index("case_id")
    extra = tables["events"].copy()
    counts = extra.groupby("case_id").size()
    eligible = counts[counts >= 5].index
    future = extra[extra["case_id"].isin(eligible)].groupby("case_id", sort=False).tail(1).copy()
    future["event_id"] = future["event_id"] + "-FUTURE"
    future["timestamp"] = pd.to_datetime(future["timestamp"]) + pd.Timedelta(days=30)
    extended = pd.concat([extra, future], ignore_index=True)
    changed = build_early_case_features(tables["cases"], extended, tables["providers"]).set_index(
        "case_id"
    )
    early_columns = CATEGORICAL_FEATURES + EARLY_NUMERIC_FEATURES
    pd.testing.assert_frame_equal(
        baseline.loc[eligible, early_columns].sort_index(),
        changed.loc[eligible, early_columns].sort_index(),
    )


def test_historical_counts_are_zero_for_earliest_case() -> None:
    tables = generate_dataset(GenerationConfig(case_count=100, seed=41))
    features = build_case_features(
        tables["cases"], tables["events"], tables["providers"]
    ).sort_values("created_at")
    earliest = features.iloc[0]
    assert earliest["provider_history_case_count"] == 0
    assert earliest["team_history_case_count"] == 0


def test_post_completion_features_are_separate() -> None:
    tables = generate_dataset(GenerationConfig(case_count=30, seed=4))
    retrospective = build_post_completion_features(tables["cases"], tables["events"])
    assert "total_duration_hours" in retrospective
    assert "final_event_count" in retrospective
    assert len(retrospective) == 30


def test_temporal_split_is_strictly_ordered() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"C-{index}" for index in range(100)],
            "created_at": pd.date_range("2025-01-01", periods=100, freq="h"),
            "target": [index % 2 for index in range(100)],
        }
    )
    splits = temporal_train_validation_test_split(frame)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (70, 15, 15)
    assert splits.train["created_at"].max() <= splits.validation["created_at"].min()
    assert splits.validation["created_at"].max() <= splits.test["created_at"].min()


def test_temporal_boundaries_report_rows_and_dates() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"C-{index}" for index in range(20)],
            "created_at": pd.date_range("2025-01-01", periods=20, freq="D"),
        }
    )
    boundaries = temporal_boundaries(temporal_train_validation_test_split(frame))
    assert boundaries["train"]["rows"] == 14
    assert boundaries["test"]["start"] > boundaries["train"]["end"]


def test_random_split_preserves_total_rows() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"C-{index}" for index in range(100)],
            "created_at": pd.date_range("2025-01-01", periods=100, freq="h"),
            "target": [index % 2 for index in range(100)],
        }
    )
    splits = random_train_validation_test_split(frame, "target", random_seed=7)
    assert len(splits.train) + len(splits.validation) + len(splits.test) == 100
    assert splits.test["target"].mean() == pytest.approx(0.5, abs=0.1)
