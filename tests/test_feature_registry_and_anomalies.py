from __future__ import annotations

import pytest

from src.data_generation.generate_claims_event_log import GenerationConfig, generate_dataset
from src.ml.feature_registry import feature_registry_frame, validate_features
from src.ml.retrospective_anomaly_detection import (
    RETROSPECTIVE_NUMERIC_FEATURES,
    train_retrospective_anomaly_model,
)


def test_feature_registry_has_task_specific_safety() -> None:
    registry = feature_registry_frame().set_index("feature_name")
    assert bool(registry.loc["queue_wait_time_total", "safe_for_sla_prediction"])
    assert not bool(registry.loc["total_cost", "safe_for_early_anomaly_detection"])
    assert bool(registry.loc["total_cost", "safe_for_retrospective_analysis"])


def test_unknown_and_post_completion_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unregistered"):
        validate_features(["invented_feature"], "sla_prediction")
    with pytest.raises(ValueError, match="Unsafe"):
        validate_features(["total_cost"], "early_anomaly_detection")


def test_retrospective_anomaly_is_explicitly_post_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.ml.retrospective_anomaly_detection as module

    monkeypatch.setattr(module, "save_model", lambda *args, **kwargs: None)
    tables = generate_dataset(GenerationConfig(case_count=100, seed=73))
    artifact, scored, metadata = train_retrospective_anomaly_model(
        tables["cases"], tables["events"], random_seed=73
    )
    assert artifact["mode"] == "post_completion"
    assert artifact["numeric_features"] == RETROSPECTIVE_NUMERIC_FEATURES
    assert scored["retrospective_anomaly_score"].between(0, 1).all()
    assert metadata["mode"] == "retrospective_anomaly_investigation"


def test_dashboard_artifact_helper_reports_missing_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.data as dashboard_data

    class TestSettings:
        data_dir = tmp_path
        model_dir = tmp_path

    monkeypatch.setattr(dashboard_data, "settings", TestSettings())
    missing = dashboard_data.missing_demo_artifacts()
    assert len(missing) == 5
    (tmp_path / "cases.csv").write_text("case_id\n")
    assert len(dashboard_data.missing_demo_artifacts()) == 4


def test_real_evidence_loader_handles_missing_and_corrupt_results(tmp_path) -> None:
    from app.real_eventlog_evidence import load_real_evidence

    evidence = load_real_evidence(tmp_path)
    assert "experiment summary" in evidence.missing
    results = tmp_path / "real_eventlog_experiments" / "results"
    results.mkdir(parents=True)
    (results / "experiment_summary.json").write_text("{not-json")
    corrupt = load_real_evidence(tmp_path)
    assert corrupt.summary is None


def test_real_evidence_leakage_table_marks_targets_unsafe() -> None:
    from app.real_eventlog_evidence import leakage_feature_table

    audit = leakage_feature_table(["first_activity"]).set_index("feature_name")
    assert bool(audit.loc["first_activity", "safe_for_prefix_prediction"])
    assert bool(audit.loc["first_activity", "used_in_model"])
    assert not bool(audit.loc["remaining_time_hours", "safe_for_prefix_prediction"])
    assert not bool(audit.loc["future_activity", "safe_for_prefix_prediction"])
