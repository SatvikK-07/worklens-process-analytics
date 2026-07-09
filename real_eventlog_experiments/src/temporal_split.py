from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ExperimentSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame) -> ExperimentSplits:
    ordered = frame.sort_values(["case_start", "case_id"])
    train_end = int(len(ordered) * 0.70)
    validation_end = int(len(ordered) * 0.85)
    splits = ExperimentSplits(
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
    if not splits.validation.empty:
        assert splits.train["case_start"].max() <= splits.validation["case_start"].min()
    if not splits.test.empty:
        assert splits.validation["case_start"].max() <= splits.test["case_start"].min()
    return splits


def random_split(frame: pd.DataFrame, target: str | None, seed: int = 42) -> ExperimentSplits:
    stratify = frame[target] if target and frame[target].value_counts().min() >= 4 else None
    train, rest = train_test_split(frame, test_size=0.30, random_state=seed, stratify=stratify)
    rest_stratify = (
        rest[target]
        if target and rest[target].nunique() == 2 and rest[target].value_counts().min() >= 2
        else None
    )
    validation, test = train_test_split(
        rest,
        test_size=0.50,
        random_state=seed,
        stratify=rest_stratify,
    )
    return ExperimentSplits(train, validation, test)
