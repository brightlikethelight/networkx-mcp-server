"""Property-based tests using Hypothesis for NetworkX MCP Server.

These tests automatically generate test cases to find edge cases and verify
fundamental properties of graph algorithms and operations.
"""

import json

import networkx as nx
import pytest
from hypothesis import given, assume, strategies as st, settings, HealthCheck

from tests.factories import (
    GraphFactory,
    graph_strategy,
    graph_with_path_strategy,
)
from networkx_mcp.core.graph_operations import GraphManager


class TestGraphPropertyInvariants:
    """Test fundamental graph properties that should always hold."""

    @given(graph_strategy(min_nodes=1, max_nodes=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_node_count_invariant(self, graph: nx.Graph):
        """Node count should equal len(nodes())."""
        assert graph.number_of_nodes() == len(list(graph.nodes()))
        assert graph.number_of_nodes() >= 0

    @given(graph_strategy(min_nodes=1, max_nodes=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_edge_count_invariant(self, graph: nx.Graph):
        """Edge count should equal len(edges())."""
        assert graph.number_of_edges() == len(list(graph.edges()))
        assert graph.number_of_edges() >= 0

    @given(graph_strategy(min_nodes=2, max_nodes=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_edge_symmetry_undirected(self, graph: nx.Graph):
        """In undirected graphs, if (u,v) is an edge, then (v,u) should also be."""
        if not graph.is_directed():
            for u, v in graph.edges():
                assert graph.has_edge(
                    v, u
                ), f"Edge ({u},{v}) exists but ({v},{u}) doesn't"

    @given(graph_strategy(min_nodes=1, max_nodes=20))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_self_loops_consistency(self, graph: nx.Graph):
        """Self-loops should be consistently reported."""
        self_loops = list(nx.selfloop_edges(graph))
        for u, v in graph.edges():
            if u == v:
                assert (u, v) in self_loops or (v, u) in self_loops

    @given(graph_strategy(min_nodes=1, max_nodes=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_degree_sum_theorem(self, graph: nx.Graph):
        """Sum of all degrees should equal 2 * number of edges."""
        if not graph.is_directed():
            degree_sum = sum(degree for node, degree in graph.degree())
            expected = 2 * graph.number_of_edges()
            assert (
                degree_sum == expected
            ), f"Degree sum {degree_sum} != 2*edges {expected}"


class TestPathfindingProperties:
    """Test properties of pathfinding algorithms."""

    @given(graph_with_path_strategy(min_nodes=3, max_nodes=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_shortest_path_exists(self, graph_and_nodes):
        """If nodes are connected, shortest path should exist."""
        graph, source, target = graph_and_nodes

        if nx.has_path(graph, source, target):
            path = nx.shortest_path(graph, source, target)
            assert path[0] == source
            assert path[-1] == target
            assert len(path) >= 2  # At least source and target

            # Path should be valid (consecutive nodes connected)
            for i in range(len(path) - 1):
                assert graph.has_edge(path[i], path[i + 1])

    @given(graph_with_path_strategy(min_nodes=3, max_nodes=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_path_length_consistency(self, graph_and_nodes):
        """Path length should equal number of edges in path."""
        graph, source, target = graph_and_nodes

        if nx.has_path(graph, source, target):
            path = nx.shortest_path(graph, source, target)
            length = nx.shortest_path_length(graph, source, target)
            assert len(path) - 1 == length

    @given(graph_strategy(min_nodes=3, max_nodes=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_path_symmetry_undirected(self, graph: nx.Graph):
        """In undirected graphs, path(u,v) length == path(v,u) length."""
        if not graph.is_directed() and graph.number_of_nodes() >= 2:
            nodes = list(graph.nodes())
            if len(nodes) >= 2:
                u, v = nodes[0], nodes[1]

                has_path_uv = nx.has_path(graph, u, v)
                has_path_vu = nx.has_path(graph, v, u)

                # Connectivity should be symmetric
                assert has_path_uv == has_path_vu

                if has_path_uv:
                    length_uv = nx.shortest_path_length(graph, u, v)
                    length_vu = nx.shortest_path_length(graph, v, u)
                    assert length_uv == length_vu


class TestCentralityProperties:
    """Test properties of centrality measures."""

    @given(graph_strategy(min_nodes=3, max_nodes=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_degree_centrality_bounds(self, graph: nx.Graph):
        """Degree centrality should be between 0 and 1."""
        if graph.number_of_nodes() > 1:
            centrality = nx.degree_centrality(graph)
            for node, value in centrality.items():
                assert (
                    0 <= value <= 1
                ), f"Degree centrality {value} out of bounds for node {node}"

    @given(graph_strategy(min_nodes=3, max_nodes=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_centrality_total_consistency(self, graph: nx.Graph):
        """Sum of degree centralities should follow expected formula."""
        if graph.number_of_nodes() > 1:
            centrality = nx.degree_centrality(graph)
            total = sum(centrality.values())

            # For undirected graphs, sum should equal 2*edges / (n*(n-1))
            if not graph.is_directed():
                n = graph.number_of_nodes()
                expected = (2 * graph.number_of_edges()) / (n * (n - 1)) if n > 1 else 0
                assert (
                    abs(total - expected) < 1e-10
                ), f"Centrality sum {total} != expected {expected}"

    @given(graph_strategy(min_nodes=2, max_nodes=8))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_betweenness_centrality_bounds(self, graph: nx.Graph):
        """Betweenness centrality should be between 0 and 1."""
        if nx.is_connected(graph) and graph.number_of_nodes() > 2:
            centrality = nx.betweenness_centrality(graph)
            for node, value in centrality.items():
                assert (
                    0 <= value <= 1
                ), f"Betweenness centrality {value} out of bounds for node {node}"


class TestConnectivityProperties:
    """Test graph connectivity properties."""

    @given(graph_strategy(min_nodes=1, max_nodes=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_connected_components_partition(self, graph: nx.Graph):
        """Connected components should partition the node set."""
        if not graph.is_directed():
            components = list(nx.connected_components(graph))

            # Components should be non-empty
            for component in components:
                assert len(component) > 0

            # Components should partition nodes (no overlap, cover all)
            all_nodes_in_components = set()
            for component in components:
                # No overlap with previous components
                assert len(all_nodes_in_components & component) == 0
                all_nodes_in_components.update(component)

            # Should cover all nodes
            assert all_nodes_in_components == set(graph.nodes())

    @given(graph_strategy(min_nodes=1, max_nodes=15))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_component_count_consistency(self, graph: nx.Graph):
        """Number of components should match component count function."""
        if not graph.is_directed():
            components = list(nx.connected_components(graph))
            count = nx.number_connected_components(graph)
            assert len(components) == count

    @given(graph_strategy(min_nodes=2, max_nodes=10, directed=True))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_strong_weak_connectivity_relationship(self, graph: nx.DiGraph):
        """If strongly connected, then weakly connected."""
        if nx.is_strongly_connected(graph):
            assert nx.is_weakly_connected(graph)


class TestGraphManagerProperties:
    """Test properties of the GraphManager class."""

    def test_graph_storage_consistency(self):
        """Graph storage and retrieval should be consistent."""
        manager = GraphManager()

        @given(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(min_codepoint=65, max_codepoint=90),
            )
        )
        @given(graph_strategy(min_nodes=1, max_nodes=10))
        @settings(suppress_health_check=[HealthCheck.too_slow])
        def check_storage(graph_id: str, graph: nx.Graph):
            # Store graph
            manager.store_graph(graph_id, graph)

            # Retrieve graph
            retrieved = manager.get_graph(graph_id)

            # Should be the same graph
            assert retrieved is not None
            assert retrieved.number_of_nodes() == graph.number_of_nodes()
            assert retrieved.number_of_edges() == graph.number_of_edges()
            assert set(retrieved.nodes()) == set(graph.nodes())
            assert set(retrieved.edges()) == set(graph.edges())

        check_storage()

    def test_graph_listing_consistency(self):
        """Graph listing should reflect stored graphs."""
        manager = GraphManager()

        # Initially empty
        assert len(manager.list_graphs()) == 0

        # Add some graphs
        g1 = GraphFactory.simple_graph(3, 2)
        g2 = GraphFactory.directed_graph(4, 3)

        manager.store_graph("graph1", g1)
        manager.store_graph("graph2", g2)

        graphs = manager.list_graphs()
        assert len(graphs) == 2
        assert "graph1" in [g["id"] for g in graphs]
        assert "graph2" in [g["id"] for g in graphs]


class TestAlgorithmProperties:
    """Test properties of graph algorithms."""

    @given(graph_strategy(min_nodes=1, max_nodes=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_density_bounds(self, graph: nx.Graph):
        """Graph density should be between 0 and 1."""
        density = nx.density(graph)
        assert 0 <= density <= 1, f"Density {density} out of bounds"

    @given(graph_strategy(min_nodes=2, max_nodes=8))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_clustering_bounds(self, graph: nx.Graph):
        """Clustering coefficients should be between 0 and 1."""
        if not graph.is_directed():
            clustering = nx.clustering(graph)
            for node, value in clustering.items():
                assert (
                    0 <= value <= 1
                ), f"Clustering {value} out of bounds for node {node}"

    @given(graph_strategy(min_nodes=3, max_nodes=8))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_transitivity_bounds(self, graph: nx.Graph):
        """Transitivity should be between 0 and 1."""
        if not graph.is_directed():
            transitivity = nx.transitivity(graph)
            assert 0 <= transitivity <= 1, f"Transitivity {transitivity} out of bounds"


class TestErrorHandlingProperties:
    """Test error handling in edge cases."""

    @given(st.text(min_size=1, max_size=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_graph_id_handling(self, invalid_id: str):
        """Operations on invalid graph IDs should fail gracefully."""
        manager = GraphManager()

        # Should return None for non-existent graphs
        result = manager.get_graph(invalid_id)
        assert result is None

    @given(graph_strategy(min_nodes=1, max_nodes=10))
    @given(st.text(min_size=1, max_size=10))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_node_handling(self, graph: nx.Graph, invalid_node: str):
        """Operations on invalid nodes should fail gracefully."""
        assume(invalid_node not in graph.nodes())

        # Should raise appropriate exceptions
        with pytest.raises((nx.NodeNotFound, KeyError)):
            nx.shortest_path(graph, invalid_node, invalid_node)

    @given(graph_strategy(min_nodes=0, max_nodes=5))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_empty_graph_handling(self, graph: nx.Graph):
        """Operations on empty or very small graphs should handle gracefully."""
        if graph.number_of_nodes() == 0:
            assert nx.density(graph) == 0
            assert len(list(nx.connected_components(graph))) == 0
        elif graph.number_of_nodes() == 1:
            assert 0 <= nx.density(graph) <= 1
            if not graph.is_directed():
                assert len(list(nx.connected_components(graph))) == 1


@pytest.mark.property
class TestMCPProtocolProperties:
    """Test MCP protocol message properties."""

    @given(st.text(min_size=1, max_size=20))
    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_tool_request_format(self, tool_name: str, arguments: dict[str, int]):
        """Tool requests should have valid JSON-RPC format."""
        from tests.factories import MCPFactory

        request = MCPFactory.tool_request(tool_name, arguments)

        # Should be valid JSON
        json_str = json.dumps(request)
        parsed = json.loads(json_str)

        # Should have required fields
        assert parsed["jsonrpc"] == "2.0"
        assert "id" in parsed
        assert parsed["method"] == "tools/call"
        assert "params" in parsed
        assert parsed["params"]["name"] == tool_name
        assert parsed["params"]["arguments"] == arguments

    @given(st.text(min_size=5, max_size=50))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_resource_uri_format(self, uri: str):
        """Resource URIs should be properly formatted."""
        from tests.factories import MCPFactory

        request = MCPFactory.resource_request(uri)

        assert request["method"] == "resources/read"
        assert request["params"]["uri"] == uri
        assert isinstance(request["id"], int)


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short"])
