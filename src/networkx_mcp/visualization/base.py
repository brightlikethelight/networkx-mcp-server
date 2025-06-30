"""Helper functions for graph visualization."""

from typing import Any

import networkx as nx


def calculate_layout(
    graph: nx.Graph, layout: str, **params: Any
) -> dict[Any, tuple[float, float]]:
    """Calculate node positions for given layout algorithm."""
    layout_funcs = {
        "spring": lambda g, **p: nx.spring_layout(g, **p),
        "circular": lambda g, **p: nx.circular_layout(g, **p),
        "random": lambda g, **p: nx.random_layout(g, **p),
        "shell": lambda g, **p: nx.shell_layout(g, **p),
        "spectral": lambda g, **p: nx.spectral_layout(g, **p),
        "planar": lambda g, **p: (
            nx.planar_layout(g, **p) if nx.is_planar(g) else nx.spring_layout(g, **p)
        ),
    }

    if layout not in layout_funcs:
        layout = "spring"  # fallback

    try:
        result: dict[Any, tuple[float, float]] = layout_funcs[layout](graph, **params)
        return result
    except Exception:
        # Fallback to spring layout if specific layout fails
        fallback_result: dict[Any, tuple[float, float]] = nx.spring_layout(graph)
        return fallback_result


def prepare_graph_data(
    graph: nx.Graph, pos: dict[Any, tuple[float, float]]
) -> dict[str, Any]:
    """Prepare graph data for visualization."""
    nodes = []
    for node in graph.nodes():
        x, y = pos.get(node, (0, 0))
        node_data = {
            "id": str(node),
            "x": float(x),
            "y": float(y),
            "degree": graph.degree(node),
            **graph.nodes[node],  # Include node attributes
        }
        nodes.append(node_data)

    edges = []
    for source, target in graph.edges():
        edge_data = {
            "source": str(source),
            "target": str(target),
            **graph.edges[source, target],  # Include edge attributes
        }
        edges.append(edge_data)

    return {
        "nodes": nodes,
        "edges": edges,
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
    }
