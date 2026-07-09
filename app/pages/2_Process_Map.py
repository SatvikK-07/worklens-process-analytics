import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from app.components.page import PLOT_CONFIG, base_layout, page_header, section_header
from app.components.styles import apply_enterprise_theme
from app.data import load_cases, load_events
from src.process_mining.path_analysis import (
    get_rework_paths,
    get_slowest_paths,
    get_top_process_paths,
)
from src.process_mining.process_graph import build_process_graph

apply_enterprise_theme()
page_header(
    "Process discovery",
    "Process map",
    "Follow the routes cases actually take, including rare transitions and rework.",
)

cases = load_cases()
events = load_events()

with st.sidebar:
    st.markdown("### Process filters")
    claim_types = st.multiselect("Claim type", sorted(cases["claim_type"].unique()), default=[])
    priorities = st.multiselect("Priority", sorted(cases["priority"].unique()), default=[])
    regions = st.multiselect("Region", sorted(cases["region"].unique()), default=[])
    outcomes = st.multiselect("Outcome", sorted(cases["outcome"].unique()), default=[])
    path_mode = st.selectbox(
        "Path view",
        ["Top 80% of transitions", "All transitions", "Rework cases", "Anomalous cases"],
    )

case_filter = cases.copy()
for column, selected in (
    ("claim_type", claim_types),
    ("priority", priorities),
    ("region", regions),
    ("outcome", outcomes),
):
    if selected:
        case_filter = case_filter[case_filter[column].isin(selected)]
if path_mode == "Rework cases":
    case_filter = case_filter[case_filter["rework_count"] > 0]
elif path_mode == "Anomalous cases":
    case_filter = case_filter[case_filter["anomaly_label"] == 1]
filtered = events[events["case_id"].isin(case_filter["case_id"])]

st.caption(f"Showing {len(case_filter):,} cases and {len(filtered):,} events")
graph = build_process_graph(filtered)
positions = nx.spring_layout(graph, seed=12, k=1.15)
edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]["frequency"], reverse=True)
if path_mode == "Top 80% of transitions" and edges:
    total_frequency = sum(edge[2]["frequency"] for edge in edges)
    selected_edges, cumulative = [], 0
    for edge in edges:
        selected_edges.append(edge)
        cumulative += edge[2]["frequency"]
        if cumulative / total_frequency >= 0.80:
            break
    edges = selected_edges

figure = go.Figure()
max_edge = max((data["frequency"] for _, _, data in edges), default=1)
for source, target, data in edges:
    x0, y0 = positions[source]
    x1, y1 = positions[target]
    figure.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(
                width=0.8 + 7 * data["frequency"] / max_edge,
                color="rgba(61, 92, 80, .32)",
            ),
            hovertemplate=(
                f"<b>{source} → {target}</b><br>"
                f"{data['frequency']:,} transitions<br>"
                f"{data['probability']:.1%} probability<br>"
                f"{data['avg_delay_hours']:.1f}h average delay<extra></extra>"
            ),
            showlegend=False,
        )
    )

nodes = list(graph.nodes(data=True))
max_frequency = max((data["frequency"] for _, data in nodes), default=1)
figure.add_trace(
    go.Scatter(
        x=[positions[node][0] for node, _ in nodes],
        y=[positions[node][1] for node, _ in nodes],
        mode="markers+text",
        text=[node for node, _ in nodes],
        textposition="top center",
        marker=dict(
            size=[20 + 34 * data["frequency"] / max_frequency for _, data in nodes],
            color=[data["avg_duration"] for _, data in nodes],
            colorscale=[[0, "#bdebd9"], [0.5, "#f4b860"], [1, "#d4513d"]],
            line=dict(color="white", width=2),
            colorbar=dict(title="Avg min", thickness=12),
        ),
        customdata=[[data["frequency"], data["avg_duration"]] for _, data in nodes],
        hovertemplate="<b>%{text}</b><br>%{customdata[0]:,} events<br>%{customdata[1]:.1f} avg min<extra></extra>",
        showlegend=False,
    )
)
figure.update_xaxes(visible=False)
figure.update_yaxes(visible=False)
figure.update_layout(dragmode="pan")
st.plotly_chart(base_layout(figure, 620), width="stretch", config=PLOT_CONFIG)

left, right = st.columns(2)
with left:
    section_header("Most common paths", "Dominant workflow variants by case volume.")
    top_paths = get_top_process_paths(filtered, 8)
    st.dataframe(
        top_paths,
        width="stretch",
        hide_index=True,
        column_config={
            "process_path": st.column_config.TextColumn("Path", width="large"),
            "case_count": st.column_config.NumberColumn("Cases", format="%d"),
            "avg_duration_hours": st.column_config.NumberColumn("Avg hours", format="%.1f"),
            "avg_manual_minutes": st.column_config.NumberColumn("Manual min", format="%.1f"),
        },
    )
with right:
    section_header("Slowest paths", "Variants with the greatest elapsed time.")
    slow_paths = get_slowest_paths(filtered, 8)
    st.dataframe(
        slow_paths,
        width="stretch",
        hide_index=True,
        column_config={
            "process_path": st.column_config.TextColumn("Path", width="large"),
            "case_count": st.column_config.NumberColumn("Cases", format="%d"),
            "avg_duration_hours": st.column_config.NumberColumn("Avg hours", format="%.1f"),
            "avg_manual_minutes": st.column_config.NumberColumn("Manual min", format="%.1f"),
        },
    )

with st.expander("Inspect highest-rework case paths"):
    st.dataframe(
        get_rework_paths(filtered, 20),
        width="stretch",
        hide_index=True,
    )
