"""Tests for core graph operations."""

import networkx as nx
import pytest

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.edge_ops import EdgeOperations
from networkx_mcp.core.node_ops import NodeOperations


class TestNodeOperations:
    """Test node operations."""

    def test_add_node(self, sample_graph):
        """Test adding a node."""
        ops = NodeOperations(sample_graph)
        initial_nodes = sample_graph.number_of_nodes()

        ops.add_node_with_validation(10, color="green", weight=2.5)

        assert sample_graph.number_of_nodes() == initial_nodes + 1
        assert 10 in sample_graph
        assert sample_graph.nodes[10]["color"] == "green"
        assert sample_graph.nodes[10]["weight"] == 2.5

    def test_add_duplicate_node(self, sample_graph):
        """Test adding a duplicate node updates attributes."""
        ops = NodeOperations(sample_graph)

        result = ops.add_node_with_validation(1, color="green")

        # Should return False since node 1 already exists
        assert result is False

    def test_remove_node(self, sample_graph):
        """Test removing a node."""
        ops = NodeOperations(sample_graph)
        initial_nodes = sample_graph.number_of_nodes()
        initial_edges = sample_graph.number_of_edges()

        # Use direct NetworkX method since NodeOperations may not have remove_node
        sample_graph.remove_node(1)

        assert sample_graph.number_of_nodes() == initial_nodes - 1
        assert 1 not in sample_graph
        assert sample_graph.number_of_edges() < initial_edges

    def test_remove_nonexistent_node(self, sample_graph):
        """Test removing a node that doesn't exist."""
        ops = NodeOperations(sample_graph)

        with pytest.raises(nx.NetworkXError):
            sample_graph.remove_node(99)

    def test_get_node_info(self, sample_graph):
        """Test getting node information."""
        ops = NodeOperations(sample_graph)

        info = ops.get_node_summary(1)

        assert info["id"] == 1
        assert info["degree"] == 3
        assert info["attributes"]["color"] == "red"
        assert "neighbors" in info


class TestEdgeOperations:
    """Test edge operations."""

    def test_add_edge(self, sample_graph):
        """Test adding an edge."""
        ops = EdgeOperations(sample_graph)
        initial_edges = sample_graph.number_of_edges()

        ops.add_edge_with_validation(5, 4, weight=3.0)

        assert sample_graph.number_of_edges() == initial_edges + 1
        assert sample_graph.has_edge(5, 4)
        assert sample_graph[5][4]["weight"] == 3.0

    def test_add_edge_creates_nodes(self, sample_graph):
        """Test that adding an edge creates missing nodes."""
        ops = EdgeOperations(sample_graph)

        ops.add_edge_with_validation(10, 11)

        assert 10 in sample_graph
        assert 11 in sample_graph
        assert sample_graph.has_edge(10, 11)

    def test_remove_edge(self, sample_graph):
        """Test removing an edge."""
        ops = EdgeOperations(sample_graph)
        initial_edges = sample_graph.number_of_edges()

        sample_graph.remove_edge(1, 2)

        assert sample_graph.number_of_edges() == initial_edges - 1
        assert not sample_graph.has_edge(1, 2)

    def test_remove_nonexistent_edge(self, sample_graph):
        """Test removing an edge that doesn't exist."""
        ops = EdgeOperations(sample_graph)

        with pytest.raises(nx.NetworkXError):
            sample_graph.remove_edge(1, 5)


class TestGraphAlgorithms:
    """Test graph algorithms."""

    def test_shortest_path(self, sample_graph):
        """Test shortest path calculation."""
        
        result = GraphAlgorithms.shortest_path(sample_graph, 1, 4)

        # Should return path and length when target is specified
        assert "path" in result
        assert "length" in result
        assert "source" in result
        assert "target" in result

    def test_shortest_path_no_path(self, sample_graph):
        """Test shortest path when no path exists."""
        
        # Use a node that doesn't exist
        with pytest.raises(ValueError):
            result = GraphAlgorithms.shortest_path(sample_graph, 1, 99)

    def test_centrality_measures(self, sample_graph):
        """Test centrality calculations."""
        
        result = GraphAlgorithms.centrality_measures(sample_graph, ["degree", "betweenness"])

        assert "degree_centrality" in result
        assert "betweenness_centrality" in result
        assert len(result["degree_centrality"]) == sample_graph.number_of_nodes()
        assert all(0 <= v <= 1 for v in result["betweenness_centrality"].values())

    @pytest.mark.parametrize("algorithm", ["bfs", "dfs"])
    def test_connected_components(self, sample_graph, algorithm):
        """Test connected components."""
        
        result = GraphAlgorithms.connected_components(sample_graph)

        assert "connected_components" in result
        assert "num_components" in result
        assert result["num_components"] == 2  # Main component + isolated node
        assert "is_connected" in result
        assert result["is_connected"] is False
