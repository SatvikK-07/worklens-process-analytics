from src.data_generation.generate_claims_event_log import (
    GenerationConfig,
    calculate_sla_breach,
    generate_dataset,
    validate_dataset,
)


def test_sla_breach_calculation() -> None:
    assert calculate_sla_breach(total_duration=80, threshold=72) == 1
    assert calculate_sla_breach(total_duration=60, threshold=72) == 0


def test_small_dataset_has_referential_integrity() -> None:
    tables = generate_dataset(GenerationConfig(case_count=250, seed=7))
    report = validate_dataset(tables, minimum_cases=250)

    assert report["case_count"] == 250
    assert report["event_count"] > report["case_count"] * 5
    assert set(tables["events"]["case_id"]).issubset(set(tables["cases"]["case_id"]))
    assert set(tables["events"]["user_id"]).issubset(set(tables["users"]["user_id"]))
