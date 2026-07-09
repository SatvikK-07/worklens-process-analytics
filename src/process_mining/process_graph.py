from __future__ import annotations

import networkx as nx
import pandas as pd

from src.process_mining.transition_matrix import calculate_transition_probabilities


def build_process_graph(events: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    activity = (
        events.groupby("activity")
        .agg(frequency=("event_id", "count"), avg_duration=("duration_minutes", "mean"))
        .reset_index()
    )
    for row in activity.itertuples(index=False):
        graph.add_node(
            row.activity,
            frequency=int(row.frequency),
            avg_duration=float(row.avg_duration),
        )
    for row in calculate_transition_probabilities(events).itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            frequency=int(row.frequency),
            probability=float(row.probability),
            avg_delay_hours=float(row.avg_transition_delay_hours),
        )
    return graph
