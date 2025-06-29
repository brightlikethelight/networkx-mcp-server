"""Test utilities and helpers."""

from typing import Any

import networkx as nx
import numpy as np


def assert_graph_equal(g1: nx.Graph, g2: nx.Graph, check_attributes: bool = True):
    """Assert two graphs are equal."""
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()
    assert set(g1.nodes()) == set(g2.nodes())
    assert set(g1.edges()) == set(g2.edges())

    if check_attributes:
        for node in g1.nodes():
            assert g1.nodes[node] == g2.nodes[node]

        for edge in g1.edges():
            assert g1.edges[edge] == g2.edges[edge]


def create_test_graph(graph_type: str, size: str = "small") -> nx.Graph:
    """Create various test graphs."""
    if size == "small":
        n = 10
    elif size == "medium":
        n = 100
    else:
        n = 1000

    if graph_type == "complete":
        return nx.complete_graph(n)
    elif graph_type == "cycle":
        return nx.cycle_graph(n)
    elif graph_type == "tree":
        return nx.balanced_tree(3, 3)
    elif graph_type == "grid":
        return nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n)))
    elif graph_type == "random":
        return nx.erdos_renyi_graph(n, 0.1)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def generate_test_data(format_type: str, graph: nx.Graph) -> Any:
    """Generate test data in various formats."""
    if format_type == "json":
        return nx.node_link_data(graph)
    elif format_type == "edgelist":
        lines = []
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            lines.append(f"{u} {v} {weight}")
        return "\n".join(lines)
    elif format_type == "adjacency":
        return nx.to_numpy_array(graph).tolist()
    else:
        raise ValueError(f"Unknown format: {format_type}")


class GraphValidator:
    """Validate graph properties."""

    @staticmethod
    def validate_connected(graph: nx.Graph) -> bool:
        """Check if graph is connected."""
        if graph.is_directed():
            return nx.is_strongly_connected(graph)
        return nx.is_connected(graph)

    @staticmethod
    def validate_tree(graph: nx.Graph) -> bool:
        """Check if graph is a tree."""
        return nx.is_tree(graph)

    @staticmethod
    def validate_bipartite(graph: nx.Graph) -> bool:
        """Check if graph is bipartite."""
        return nx.is_bipartite(graph)

    @staticmethod
    def validate_planar(graph: nx.Graph) -> bool:
        """Check if graph is planar."""
        return nx.is_planar(graph)
