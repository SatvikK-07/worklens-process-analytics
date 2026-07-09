from __future__ import annotations

import pandas as pd

from src.analytics.assistant_engine import answer_business_question
from src.analytics.automation_roi_engine import (
    calculate_manual_hours,
    estimate_annual_savings,
    rank_automation_candidates,
)
from src.analytics.bottleneck_engine import calculate_wait_times, rank_bottlenecks
from src.analytics.data_quality_engine import run_data_quality_checks
from src.analytics.kpi_engine import calculate_kpis, monthly_trends
from src.data_generation.generate_claims_event_log import GenerationConfig, generate_dataset
from src.process_mining.path_analysis import (
    get_rare_paths,
    get_rework_paths,
    get_slowest_paths,
    get_top_process_paths,
)
from src.process_mining.process_graph import build_process_graph


def small_tables() -> dict[str, pd.DataFrame]:
    return generate_dataset(GenerationConfig(case_count=180, seed=21))


def test_wait_times_are_non_negative() -> None:
    events = calculate_wait_times(small_tables()["events"])
    assert events["wait_hours"].ge(0).all()


def test_full_bottleneck_ranking_has_score() -> None:
    tables = small_tables()
    ranked = rank_bottlenecks(tables["events"], tables["cases"])
    assert ranked["bottleneck_score"].between(0, 100).all()
    assert ranked.iloc[0]["bottleneck_score"] >= ranked.iloc[-1]["bottleneck_score"]


def test_automation_ranking_and_helpers() -> None:
    tables = small_tables()
    ranked = rank_automation_candidates(tables["events"], tables["cases"])
    assert ranked["estimated_monthly_savings"].gt(0).all()
    assert calculate_manual_hours(100, 30) == 50
    assert estimate_annual_savings(1_000) == 12_000


def test_kpi_engine_and_monthly_trends() -> None:
    tables = small_tables()
    candidates = rank_automation_candidates(tables["events"], tables["cases"])
    bottlenecks = rank_bottlenecks(tables["events"], tables["cases"])
    kpis = calculate_kpis(tables["cases"], bottlenecks, candidates)
    trends = monthly_trends(tables["cases"])
    assert kpis["total_cases"] == 180
    assert kpis["automation_savings"] > 0
    assert not trends.empty


def test_data_quality_detects_negative_and_duplicate_events() -> None:
    tables = small_tables()
    events = tables["events"].copy()
    events.loc[0, "duration_minutes"] = -1
    events.loc[1, "event_id"] = events.loc[0, "event_id"]
    score, issues = run_data_quality_checks(tables["cases"], events, tables["users"])
    assert score < 100
    assert {"Negative duration", "Duplicate event"}.issubset(set(issues["issue_type"]))


def test_process_path_variants_and_graph() -> None:
    events = small_tables()["events"]
    assert not get_top_process_paths(events, 5).empty
    assert not get_slowest_paths(events, 5).empty
    assert not get_rework_paths(events, 5).empty
    assert not get_rare_paths(events, 5).empty
    graph = build_process_graph(events)
    assert graph.number_of_nodes() >= 8
    assert graph.number_of_edges() >= 8


def test_assistant_answers_all_governed_intents() -> None:
    tables = small_tables()
    candidates = rank_automation_candidates(tables["events"], tables["cases"])
    questions = [
        "Which activity should we automate first?",
        "Why are prior authorization cases delayed?",
        "Show top breach risk cases",
        "Which provider causes rework?",
        "Tell me something else",
    ]
    for question in questions:
        answer, _ = answer_business_question(
            question, tables["cases"], candidates, tables["providers"]
        )
        assert answer
