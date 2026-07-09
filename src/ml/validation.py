from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_train_validation_test_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DataSplits:
    """Split chronologically so evaluation always occurs on future cases."""
    ordered = frame.assign(created_at=pd.to_datetime(frame["created_at"])).sort_values(
        ["created_at", "case_id"]
    )
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    splits = DataSplits(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )
    if not splits.validation.empty and not splits.train.empty:
        assert splits.train["created_at"].max() <= splits.validation["created_at"].min()
    if not splits.test.empty and not splits.validation.empty:
        assert splits.validation["created_at"].max() <= splits.test["created_at"].min()
    return splits


def random_train_validation_test_split(
    frame: pd.DataFrame,
    target: str | None,
    random_seed: int = 42,
) -> DataSplits:
    """Create a reproducible random 70/15/15 comparison split."""
    stratify = frame[target] if target else None
    train, remaining = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_seed,
        stratify=stratify,
    )
    remaining_stratify = remaining[target] if target else None
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=random_seed,
        stratify=remaining_stratify,
    )
    return DataSplits(train=train, validation=validation, test=test)


def temporal_boundaries(splits: DataSplits) -> dict[str, dict[str, str | int]]:
    result = {}
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        dates = pd.to_datetime(frame["created_at"])
        result[name] = {
            "rows": len(frame),
            "start": dates.min().isoformat(),
            "end": dates.max().isoformat(),
        }
    return result
