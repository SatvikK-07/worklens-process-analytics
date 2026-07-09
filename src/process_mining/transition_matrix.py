from __future__ import annotations

import pandas as pd


def _with_next_activity(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["case_id", "timestamp", "event_id"])
    frame["next_activity"] = frame.groupby("case_id")["activity"].shift(-1)
    frame["next_timestamp"] = frame.groupby("case_id")["timestamp"].shift(-1)
    frame["transition_delay_hours"] = (
        frame["next_timestamp"] - frame["timestamp"]
    ).dt.total_seconds() / 3600
    return frame.dropna(subset=["next_activity"])


def build_transition_matrix(events: pd.DataFrame) -> pd.DataFrame:
    frame = _with_next_activity(events)
    return pd.crosstab(frame["activity"], frame["next_activity"])


def calculate_transition_probabilities(events: pd.DataFrame) -> pd.DataFrame:
    frame = _with_next_activity(events)
    transitions = (
        frame.groupby(["activity", "next_activity"], as_index=False)
        .agg(
            frequency=("case_id", "count"),
            avg_transition_delay_hours=("transition_delay_hours", "mean"),
        )
        .rename(columns={"activity": "source", "next_activity": "target"})
    )
    transitions["probability"] = transitions["frequency"] / transitions.groupby("source")[
        "frequency"
    ].transform("sum")
    return transitions.sort_values("frequency", ascending=False, ignore_index=True)
