import sqlite3
from pathlib import Path

from src.data_generation.generate_claims_event_log import (
    GenerationConfig,
    generate_dataset,
    write_dataset,
)
from src.ingestion.load_to_db import load_database


def test_database_loader_creates_views(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    database_path = tmp_path / "test.db"
    write_dataset(
        generate_dataset(GenerationConfig(case_count=100, seed=11)),
        data_dir,
    )
    counts = load_database(data_dir, database_path)

    assert counts["cases"] == 100
    with sqlite3.connect(database_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        activity_rows = connection.execute(
            "SELECT COUNT(*) FROM activity_performance_view"
        ).fetchone()[0]
    assert total == 100
    assert activity_rows >= 8
