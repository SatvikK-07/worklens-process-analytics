from __future__ import annotations

from pathlib import Path

from scripts.check_repo_hygiene import find_hygiene_issues


def issue_paths(root: Path) -> set[str]:
    return {str(issue.path) for issue in find_hygiene_issues(root)}


def test_hygiene_checker_detects_cache_database_and_model_binary(tmp_path: Path) -> None:
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "module.pyc").write_bytes(b"compiled")
    (tmp_path / "local.db").write_bytes(b"sqlite")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "sample.pkl").write_bytes(b"model")
    paths = issue_paths(tmp_path)
    assert "__pycache__" in paths
    assert "__pycache__/module.pyc" in paths
    assert "local.db" in paths
    assert "models/sample.pkl" in paths


def test_hygiene_checker_allows_small_committed_samples(tmp_path: Path) -> None:
    sample = tmp_path / "data" / "sample"
    sample.mkdir(parents=True)
    (sample / "events.csv").write_text("case_id,activity\nC-1,Start\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n")
    assert find_hygiene_issues(tmp_path) == []


def test_hygiene_checker_rejects_large_csv_outside_sample(tmp_path: Path, monkeypatch) -> None:
    import scripts.check_repo_hygiene as hygiene

    monkeypatch.setattr(hygiene, "MAX_NON_SAMPLE_CSV_BYTES", 10)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "generated.csv").write_text("more than ten bytes")
    assert "reports/generated.csv" in issue_paths(tmp_path)
