import pandas as pd

from src.analytics.automation_roi_engine import estimate_monthly_savings
from src.analytics.bottleneck_engine import calculate_bottleneck_score
from src.analytics.rework_engine import detect_rework_loops
from src.process_mining.transition_matrix import build_transition_matrix


def test_bottleneck_score_ranks_slowest_activity() -> None:
    metrics = pd.DataFrame(
        {
            "activity": ["Fast", "Slow"],
            "avg_duration_minutes": [1, 10],
            "p95_duration_minutes": [2, 20],
            "avg_wait_hours": [0.1, 5],
            "sla_breach_contribution": [0.01, 0.5],
            "rework_rate": [0.01, 0.4],
        }
    )
    ranked = calculate_bottleneck_score(metrics)
    assert ranked.loc[ranked["activity"] == "Slow", "bottleneck_score"].iloc[0] == 100


def test_rework_loop_detection() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["1", "2", "3"],
            "case_id": ["C-1"] * 3,
            "activity": ["Document Review", "Provider Clarification", "Document Review"],
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "duration_minutes": [10, 20, 10],
            "team": ["Claims", "Provider Services", "Claims"],
        }
    )
    loops = detect_rework_loops(events)
    assert loops.iloc[0]["cases_affected"] == 1
    assert loops.iloc[0]["occurrences"] == 1


def test_automation_savings_calculation() -> None:
    result = estimate_monthly_savings(42_000, 4.2, 35, 0.70)
    assert round(result, 2) == 72_030.00


def test_transition_matrix_counts_edges() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["1", "2", "3", "4"],
            "case_id": ["A", "A", "B", "B"],
            "activity": ["Start", "End", "Start", "End"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        }
    )
    matrix = build_transition_matrix(events)
    assert matrix.loc["Start", "End"] == 2
