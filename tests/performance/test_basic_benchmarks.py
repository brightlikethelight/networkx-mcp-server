"""Basic performance benchmarks for NetworkX MCP Server.

This module provides basic benchmark tests to ensure CI benchmarks workflow
doesn't fail due to missing test files.
"""

import networkx as nx
import pytest

from networkx_mcp.core.basic_operations import add_edges, add_nodes, create_graph


def test_graph_creation_benchmark(benchmark):
    """Benchmark basic graph creation performance."""

    def create_test_graph():
        graphs = {}
        result = create_graph("test_graph", directed=False, graphs=graphs)
        return result, graphs

    result, graphs = benchmark(create_test_graph)
    assert result["created"] is True
    assert "test_graph" in graphs


def test_node_addition_benchmark(benchmark):
    """Benchmark node addition performance."""

    def add_test_nodes():
        graphs = {}
        create_graph("bench_graph", directed=False, graphs=graphs)
        nodes = [f"node_{i}" for i in range(100)]
        return add_nodes("bench_graph", nodes, graphs=graphs)

    result = benchmark(add_test_nodes)
    assert result["nodes_added"] == 100


def test_edge_addition_benchmark(benchmark):
    """Benchmark edge addition performance."""

    def add_test_edges():
        graphs = {}
        create_graph("edge_graph", directed=False, graphs=graphs)

        # Create test nodes first
        nodes = [f"node_{i}" for i in range(50)]
        add_nodes("edge_graph", nodes, graphs=graphs)

        # Create edges between consecutive nodes
        edges = [[f"node_{i}", f"node_{i + 1}"] for i in range(49)]
        return add_edges("edge_graph", edges, graphs=graphs)

    result = benchmark(add_test_edges)
    assert result["edges_added"] == 49


def test_networkx_algorithms_benchmark(benchmark):
    """Benchmark core NetworkX algorithms."""

    def run_algorithms():
        # Create a simple graph for benchmarking
        G = nx.erdos_renyi_graph(100, 0.1)

        # Run basic algorithms
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, k=10)  # Sample for speed

        return len(degree_centrality), len(betweenness)

    degree_len, betweenness_len = benchmark(run_algorithms)
    assert degree_len == 100
    assert betweenness_len == 100


if __name__ == "__main__":
    # Allow running benchmarks directly
    pytest.main([__file__, "--benchmark-only", "-v"])
