from src.analytics.simulation_engine import simulate_operating_model


def test_simulation_improves_duration_and_breach_rate() -> None:
    result = simulate_operating_model(
        baseline_avg_hours=40,
        baseline_breach_rate=0.20,
        automation_hours_per_case=2,
        automation_coverage=0.7,
        medical_director_delay_share=0.25,
        capacity_increase=0.2,
        rework_hours_per_case=3,
        rework_reduction=0.3,
    )
    assert result["new_avg_hours"] < 40
    assert result["new_breach_rate"] < 0.20
