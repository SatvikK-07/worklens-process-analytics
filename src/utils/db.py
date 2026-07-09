from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.ingestion.load_to_db import sqlite_path_from_url
from src.utils.config import settings


def database_path() -> Path:
    return sqlite_path_from_url(settings.database_url)


@contextmanager
def connect(read_only: bool = False) -> Iterator[sqlite3.Connection]:
    path = database_path()
    if read_only:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def query_df(sql: str, params: Sequence[object] | None = None) -> pd.DataFrame:
    with connect(read_only=True) as connection:
        return pd.read_sql_query(sql, connection, params=params or ())


def table_exists(name: str) -> bool:
    if not database_path().exists():
        return False
    with connect(read_only=True) as connection:
        result = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (name,),
        ).fetchone()
    return result is not None
