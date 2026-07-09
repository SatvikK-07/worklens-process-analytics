from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config import settings
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TABLE_ORDER = ("users", "providers", "cases", "events")


def sqlite_path_from_url(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("The local loader currently requires a sqlite:/// URL.")
    return Path(database_url.removeprefix(prefix)).expanduser().resolve()


def load_database(data_dir: Path, database_path: Path, reset: bool = True) -> dict[str, int]:
    """Load generated CSVs, indexes, and views into a transactional SQLite database."""
    schema_path = ROOT / "database" / "schema.sql"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and database_path.exists():
        database_path.unlink()

    counts: dict[str, int] = {}
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        schema = schema_path.read_text()
        table_ddl, view_ddl = schema.split("CREATE VIEW case_summary_view", maxsplit=1)
        connection.executescript(table_ddl)

        for table in TABLE_ORDER:
            csv_path = data_dir / f"{table}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing generated data: {csv_path}")
            if table == "events":
                count = 0
                for chunk in pd.read_csv(csv_path, chunksize=75_000):
                    chunk.to_sql(table, connection, if_exists="append", index=False)
                    count += len(chunk)
                counts[table] = count
            else:
                frame = pd.read_csv(csv_path)
                frame.to_sql(table, connection, if_exists="append", index=False)
                counts[table] = len(frame)

        connection.executescript("CREATE VIEW case_summary_view" + view_ddl)
        connection.execute("PRAGMA foreign_keys = ON")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"Foreign-key validation failed: {violations[:5]}")
        connection.commit()

    LOGGER.info("Loaded %s into %s", counts, database_path)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load WorkLens synthetic data into SQLite.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument(
        "--database",
        type=Path,
        default=sqlite_path_from_url(settings.database_url),
    )
    parser.add_argument("--no-reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_database(args.data_dir, args.database, reset=not args.no_reset)


if __name__ == "__main__":
    main()
