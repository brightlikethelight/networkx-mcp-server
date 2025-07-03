"""Test factories for generating test data using Factory Boy and Hypothesis.

This module provides comprehensive test data generation for NetworkX graphs,
MCP server testing, and property-based testing scenarios.
"""

import random
from typing import Any

import networkx as nx
from hypothesis import strategies as st
from hypothesis.strategies import composite


class GraphFactory:
    """Factory for creating various types of NetworkX graphs for testing."""

    @staticmethod
    def simple_graph(n_nodes: int = 5, n_edges: int = 6) -> nx.Graph:
        """Create a simple undirected graph."""
        G = nx.Graph()
        nodes = list(range(n_nodes))
        G.add_nodes_from(nodes)

        # Add random edges
        edges_added = 0
        attempts = 0
        max_attempts = n_edges * 10

        while edges_added < n_edges and attempts < max_attempts:
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                edges_added += 1
            attempts += 1

        return G

    @staticmethod
    def directed_graph(n_nodes: int = 5, n_edges: int = 6) -> nx.DiGraph:
        """Create a simple directed graph."""
        G = nx.DiGraph()
        nodes = list(range(n_nodes))
        G.add_nodes_from(nodes)

        # Add random directed edges
        edges_added = 0
        attempts = 0
        max_attempts = n_edges * 10

        while edges_added < n_edges and attempts < max_attempts:
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                edges_added += 1
            attempts += 1

        return G

    @staticmethod
    def weighted_graph(n_nodes: int = 5, n_edges: int = 6) -> nx.Graph:
        """Create a weighted graph."""
        G = GraphFactory.simple_graph(n_nodes, n_edges)
        for u, v in G.edges():
            G[u][v]["weight"] = random.uniform(0.1, 10.0)
        return G

    @staticmethod
    def complete_graph(n_nodes: int = 5) -> nx.Graph:
        """Create a complete graph."""
        return nx.complete_graph(n_nodes)

    @staticmethod
    def tree_graph(n_nodes: int = 7) -> nx.Graph:
        """Create a tree graph."""
        # Create a path graph as a simple tree
        return nx.path_graph(n_nodes)

    @staticmethod
    def bipartite_graph(n_left: int = 3, n_right: int = 4, p: float = 0.5) -> nx.Graph:
        """Create a bipartite graph."""
        return nx.bipartite.random_graph(n_left, n_right, p, seed=42)

    @staticmethod
    def social_network(n_nodes: int = 20, k: int = 4, p: float = 0.3) -> nx.Graph:
        """Create a social network-like graph using Watts-Strogatz model."""
        return nx.watts_strogatz_graph(n_nodes, k, p, seed=42)

    @staticmethod
    def scale_free_graph(n_nodes: int = 20) -> nx.Graph:
        """Create a scale-free graph using Barabási-Albert model."""
        return nx.barabasi_albert_graph(n_nodes, 2, seed=42)

    @staticmethod
    def graph_with_attributes(n_nodes: int = 5) -> nx.Graph:
        """Create a graph with node and edge attributes."""
        G = GraphFactory.simple_graph(n_nodes)

        # Add node attributes
        for node in G.nodes():
            G.nodes[node]["label"] = f"Node_{node}"
            G.nodes[node]["value"] = random.randint(1, 100)
            G.nodes[node]["category"] = random.choice(["A", "B", "C"])

        # Add edge attributes
        for u, v in G.edges():
            G[u][v]["weight"] = random.uniform(0.1, 10.0)
            G[u][v]["type"] = random.choice(["friend", "colleague", "family"])

        return G

    @staticmethod
    def disconnected_graph(
        n_components: int = 3, nodes_per_component: int = 3
    ) -> nx.Graph:
        """Create a disconnected graph with multiple components."""
        G = nx.Graph()
        node_id = 0

        for _ in range(n_components):
            # Create a small connected component
            component_nodes = list(range(node_id, node_id + nodes_per_component))
            G.add_nodes_from(component_nodes)

            # Connect nodes in a path to ensure connectivity
            for i in range(len(component_nodes) - 1):
                G.add_edge(component_nodes[i], component_nodes[i + 1])

            node_id += nodes_per_component

        return G


# Hypothesis strategies for property-based testing
@composite
def graph_strategy(
    draw,
    min_nodes: int = 1,
    max_nodes: int = 20,
    min_edges: int = 0,
    max_edges: int | None = None,
    directed: bool = False,
) -> nx.Graph:
    """Generate random graphs using Hypothesis."""
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    if max_edges is None:
        if directed:
            max_possible_edges = n_nodes * (n_nodes - 1)
        else:
            max_possible_edges = n_nodes * (n_nodes - 1) // 2
        max_edges = min(max_possible_edges, max_nodes * 3)

    n_edges = draw(st.integers(min_value=min_edges, max_value=max_edges))

    if directed:
        return GraphFactory.directed_graph(n_nodes, n_edges)
    else:
        return GraphFactory.simple_graph(n_nodes, n_edges)


@composite
def node_strategy(draw, graph: nx.Graph) -> Any:
    """Generate a valid node from a graph."""
    if graph.number_of_nodes() == 0:
        return None
    return draw(st.sampled_from(list(graph.nodes())))


@composite
def edge_strategy(draw, graph: nx.Graph) -> tuple[Any, Any] | None:
    """Generate a valid edge from a graph."""
    if graph.number_of_edges() == 0:
        return None
    return draw(st.sampled_from(list(graph.edges())))


@composite
def graph_with_path_strategy(
    draw, min_nodes: int = 3, max_nodes: int = 10
) -> tuple[nx.Graph, Any, Any]:
    """Generate a graph guaranteed to have a path between two specific nodes."""
    n_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    # Create a path graph to ensure connectivity
    G = nx.path_graph(n_nodes)

    # Add some random edges
    n_extra_edges = draw(st.integers(min_value=0, max_value=n_nodes))
    for _ in range(n_extra_edges):
        u = draw(st.integers(min_value=0, max_value=n_nodes - 1))
        v = draw(st.integers(min_value=0, max_value=n_nodes - 1))
        if u != v:
            G.add_edge(u, v)

    # Select source and target
    source = 0
    target = n_nodes - 1

    return G, source, target


# MCP Request/Response factories
class MCPFactory:
    """Factory for creating MCP protocol messages."""

    @staticmethod
    def tool_request(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a tool request message."""
        return {
            "jsonrpc": "2.0",
            "id": random.randint(1, 10000),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

    @staticmethod
    def resource_request(uri: str) -> dict[str, Any]:
        """Create a resource request message."""
        return {
            "jsonrpc": "2.0",
            "id": random.randint(1, 10000),
            "method": "resources/read",
            "params": {"uri": uri},
        }

    @staticmethod
    def prompt_request(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create a prompt request message."""
        return {
            "jsonrpc": "2.0",
            "id": random.randint(1, 10000),
            "method": "prompts/get",
            "params": {"name": name, "arguments": arguments},
        }


# Data generators for testing different graph properties
class DataGenerators:
    """Generators for specific test data patterns."""

    @staticmethod
    def centrality_test_graphs() -> list[tuple[str, nx.Graph]]:
        """Generate graphs for testing centrality algorithms."""
        return [
            ("star", nx.star_graph(5)),
            ("path", nx.path_graph(6)),
            ("cycle", nx.cycle_graph(6)),
            ("complete", nx.complete_graph(5)),
            ("wheel", nx.wheel_graph(6)),
        ]

    @staticmethod
    def community_test_graphs() -> list[tuple[str, nx.Graph]]:
        """Generate graphs for testing community detection."""
        graphs = []

        # Two connected cliques
        G1 = nx.Graph()
        G1.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle 1
        G1.add_edges_from([(3, 4), (4, 5), (5, 3)])  # Triangle 2
        G1.add_edge(0, 3)  # Bridge
        graphs.append(("two_triangles", G1))

        # Karate club graph
        graphs.append(("karate", nx.karate_club_graph()))

        return graphs

    @staticmethod
    def pathfinding_test_graphs() -> list[tuple[str, nx.Graph, Any, Any]]:
        """Generate graphs for testing pathfinding algorithms."""
        return [
            ("simple_path", nx.path_graph(5), 0, 4),
            ("cycle_with_shortcut", nx.cycle_graph(6), 0, 3),
            ("weighted_grid", nx.grid_2d_graph(3, 3), (0, 0), (2, 2)),
        ]

    @staticmethod
    def invalid_inputs() -> list[tuple[str, Any]]:
        """Generate invalid inputs for error testing."""
        return [
            ("empty_string", ""),
            ("none", None),
            ("negative_int", -1),
            ("invalid_node", "nonexistent_node"),
            ("empty_list", []),
            ("wrong_type", {"not": "a_graph"}),
        ]


# Security test data
class SecurityTestData:
    """Generate data for security boundary testing."""

    @staticmethod
    def malicious_graph_ids() -> list[str]:
        """Generate potentially malicious graph IDs."""
        return [
            "../../../etc/passwd",
            "../../../../windows/system32/config/sam",
            "<script>alert('xss')</script>",
            "'; DROP TABLE graphs; --",
            "\\x00\\x01\\x02",
            "a" * 10000,  # Very long string
            "../../.env",
            "/dev/null",
            "CON",
            "PRN",
            "AUX",  # Windows reserved names
        ]

    @staticmethod
    def large_inputs() -> dict[str, Any]:
        """Generate large inputs to test DoS protection."""
        return {
            "huge_node_list": list(range(100000)),
            "deep_nesting": {"a": {"b": {"c": {"d": "e" * 1000}}}},
            "large_string": "x" * 1000000,
            "many_edges": [(i, i + 1) for i in range(50000)],
        }


__all__ = [
    "GraphFactory",
    "MCPFactory",
    "DataGenerators",
    "SecurityTestData",
    "graph_strategy",
    "node_strategy",
    "edge_strategy",
    "graph_with_path_strategy",
]
