from __future__ import annotations


def simulate_operating_model(
    baseline_avg_hours: float,
    baseline_breach_rate: float,
    automation_hours_per_case: float,
    automation_coverage: float,
    medical_director_delay_share: float,
    capacity_increase: float,
    rework_hours_per_case: float,
    rework_reduction: float,
) -> dict[str, float]:
    automation_gain = automation_hours_per_case * automation_coverage
    capacity_gain = (
        baseline_avg_hours
        * medical_director_delay_share
        * capacity_increase
        / (1 + capacity_increase)
    )
    rework_gain = rework_hours_per_case * rework_reduction
    new_avg = max(0.1, baseline_avg_hours - automation_gain - capacity_gain - rework_gain)
    ratio = new_avg / baseline_avg_hours
    new_breach_rate = max(0.0, min(1.0, baseline_breach_rate * ratio**1.35))
    return {
        "new_avg_hours": new_avg,
        "hours_reduced": baseline_avg_hours - new_avg,
        "new_breach_rate": new_breach_rate,
        "breach_rate_reduction": baseline_breach_rate - new_breach_rate,
    }
