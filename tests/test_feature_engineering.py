from src.data_generation.generate_claims_event_log import GenerationConfig, generate_dataset
from src.ml.feature_engineering import MODEL_FEATURES, build_case_features


def test_feature_engineering_returns_one_row_per_case() -> None:
    tables = generate_dataset(GenerationConfig(case_count=120, seed=19))
    features = build_case_features(tables["cases"], tables["events"], tables["providers"])
    assert len(features) == 120
    assert set(MODEL_FEATURES).issubset(features.columns)
    assert features["queue_wait_time_total"].ge(0).all()
