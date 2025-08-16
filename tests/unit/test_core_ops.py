"""Comprehensive tests for core/edge_ops.py and core/node_ops.py - Target: 100% coverage.

These modules are critical for graph operations and must be thoroughly tested.
"""

import networkx as nx
import pytest

from networkx_mcp.core.edge_ops import EdgeOperations
from networkx_mcp.core.node_ops import NodeOperations


class TestEdgeOperations:
    """Test EdgeOperations class."""

    def test_init(self):
        """Test EdgeOperations initialization."""
        graph = nx.Graph()
        edge_ops = EdgeOperations(graph)
        assert edge_ops.graph is graph

    def test_add_edge_with_validation_new_edge(self):
        """Test adding a new edge with validation."""
        graph = nx.Graph()
        edge_ops = EdgeOperations(graph)

        # Add new edge
        result = edge_ops.add_edge_with_validation(1, 2, weight=1.5, color="red")
        assert result is True
        assert graph.has_edge(1, 2)
        assert graph.edges[1, 2]["weight"] == 1.5
        assert graph.edges[1, 2]["color"] == "red"

    def test_add_edge_with_validation_existing_edge(self):
        """Test adding an existing edge returns False."""
        graph = nx.Graph()
        graph.add_edge(1, 2)
        edge_ops = EdgeOperations(graph)

        # Try to add existing edge
        result = edge_ops.add_edge_with_validation(1, 2, weight=2.0)
        assert result is False
        # Original edge should remain unchanged
        assert "weight" not in graph.edges[1, 2]

    def test_add_edge_with_validation_self_loop(self):
        """Test adding self-loop edge."""
        graph = nx.Graph()
        edge_ops = EdgeOperations(graph)

        result = edge_ops.add_edge_with_validation(1, 1, type="self_loop")
        assert result is True
        assert graph.has_edge(1, 1)
        assert graph.edges[1, 1]["type"] == "self_loop"

    def test_bulk_add_edges_empty_graph(self):
        """Test bulk adding edges to empty graph."""
        graph = nx.Graph()
        edge_ops = EdgeOperations(graph)

        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        count = edge_ops.bulk_add_edges(edges)

        assert count == 4
        assert graph.number_of_edges() == 4
        for edge in edges:
            assert graph.has_edge(*edge)

    def test_bulk_add_edges_with_attributes(self):
        """Test bulk adding edges with attributes."""
        graph = nx.Graph()
        edge_ops = EdgeOperations(graph)

        edges = [(1, 2, {"weight": 1.0}), (2, 3, {"weight": 2.0})]
        count = edge_ops.bulk_add_edges(edges)

        assert count == 2
        assert graph.edges[1, 2]["weight"] == 1.0
        assert graph.edges[2, 3]["weight"] == 2.0

    def test_bulk_add_edges_partial_existing(self):
        """Test bulk adding where some edges already exist."""
        graph = nx.Graph()
        graph.add_edge(1, 2)
        edge_ops = EdgeOperations(graph)

        edges = [(1, 2), (2, 3), (3, 4)]  # First edge already exists
        count = edge_ops.bulk_add_edges(edges)

        assert count == 2  # Only 2 new edges added
        assert graph.number_of_edges() == 3

    def test_bulk_add_edges_empty_list(self):
        """Test bulk adding empty list of edges."""
        graph = nx.Graph()
        edge_ops = EdgeOperations(graph)

        count = edge_ops.bulk_add_edges([])
        assert count == 0
        assert graph.number_of_edges() == 0

    def test_get_edge_summary_existing_edge(self):
        """Test getting summary of existing edge."""
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1.5, color="blue", label="test")
        edge_ops = EdgeOperations(graph)

        summary = edge_ops.get_edge_summary(1, 2)

        assert summary["source"] == 1
        assert summary["target"] == 2
        assert summary["weight"] == 1.5
        assert summary["attributes"]["weight"] == 1.5
        assert summary["attributes"]["color"] == "blue"
        assert summary["attributes"]["label"] == "test"

    def test_get_edge_summary_no_weight(self):
        """Test edge summary with no weight attribute."""
        graph = nx.Graph()
        graph.add_edge("A", "B", color="red")
        edge_ops = EdgeOperations(graph)

        summary = edge_ops.get_edge_summary("A", "B")

        assert summary["weight"] == 1  # Default weight
        assert summary["attributes"]["color"] == "red"
        assert "weight" not in summary["attributes"]

    def test_get_edge_summary_nonexistent_edge(self):
        """Test getting summary of non-existent edge."""
        graph = nx.Graph()
        graph.add_edge(1, 2)
        edge_ops = EdgeOperations(graph)

        with pytest.raises(ValueError) as exc:
            edge_ops.get_edge_summary(1, 3)
        assert "Edge (1, 3) not found" in str(exc.value)

    def test_get_edge_summary_directed_graph(self):
        """Test edge summary with directed graph."""
        graph = nx.DiGraph()
        graph.add_edge(1, 2, weight=3.0)
        edge_ops = EdgeOperations(graph)

        summary = edge_ops.get_edge_summary(1, 2)
        assert summary["source"] == 1
        assert summary["target"] == 2
        assert summary["weight"] == 3.0

        # Reverse edge should not exist
        with pytest.raises(ValueError):
            edge_ops.get_edge_summary(2, 1)


class TestNodeOperations:
    """Test NodeOperations class."""

    def test_init(self):
        """Test NodeOperations initialization."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)
        assert node_ops.graph is graph

    def test_add_node_with_validation_new_node(self):
        """Test adding a new node with validation."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)

        result = node_ops.add_node_with_validation("node1", color="red", size=10)
        assert result is True
        assert "node1" in graph
        assert graph.nodes["node1"]["color"] == "red"
        assert graph.nodes["node1"]["size"] == 10

    def test_add_node_with_validation_existing_node(self):
        """Test adding an existing node returns False."""
        graph = nx.Graph()
        graph.add_node("node1", original=True)
        node_ops = NodeOperations(graph)

        result = node_ops.add_node_with_validation("node1", new=True)
        assert result is False
        # Original attributes should remain
        assert graph.nodes["node1"]["original"] is True
        assert "new" not in graph.nodes["node1"]

    def test_add_node_with_validation_integer_id(self):
        """Test adding node with integer ID."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)

        result = node_ops.add_node_with_validation(42, label="answer")
        assert result is True
        assert 42 in graph
        assert graph.nodes[42]["label"] == "answer"

    def test_bulk_add_nodes_simple(self):
        """Test bulk adding simple node IDs."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)

        nodes = [1, 2, 3, "A", "B"]
        count = node_ops.bulk_add_nodes(nodes)

        assert count == 5
        assert graph.number_of_nodes() == 5
        for node in nodes:
            assert node in graph

    def test_bulk_add_nodes_with_attributes(self):
        """Test bulk adding nodes with attributes."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)

        nodes = [(1, {"color": "red"}), (2, {"color": "blue"}), (3, {"color": "green"})]
        count = node_ops.bulk_add_nodes(nodes)

        assert count == 3
        assert graph.nodes[1]["color"] == "red"
        assert graph.nodes[2]["color"] == "blue"
        assert graph.nodes[3]["color"] == "green"

    def test_bulk_add_nodes_partial_existing(self):
        """Test bulk adding where some nodes already exist."""
        graph = nx.Graph()
        graph.add_nodes_from([1, 2])
        node_ops = NodeOperations(graph)

        nodes = [1, 2, 3, 4]  # First two already exist
        count = node_ops.bulk_add_nodes(nodes)

        assert count == 2  # Only 2 new nodes added
        assert graph.number_of_nodes() == 4

    def test_bulk_add_nodes_empty_list(self):
        """Test bulk adding empty list of nodes."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)

        count = node_ops.bulk_add_nodes([])
        assert count == 0
        assert graph.number_of_nodes() == 0

    def test_get_node_summary_existing_node(self):
        """Test getting summary of existing node."""
        graph = nx.Graph()
        graph.add_node(1, color="red", size=5)
        graph.add_edges_from([(1, 2), (1, 3), (1, 4)])
        node_ops = NodeOperations(graph)

        summary = node_ops.get_node_summary(1)

        assert summary["id"] == 1
        assert summary["attributes"]["color"] == "red"
        assert summary["attributes"]["size"] == 5
        assert summary["degree"] == 3
        assert set(summary["neighbors"]) == {2, 3, 4}

    def test_get_node_summary_isolated_node(self):
        """Test summary of isolated node (no edges)."""
        graph = nx.Graph()
        graph.add_node("isolated", label="alone")
        node_ops = NodeOperations(graph)

        summary = node_ops.get_node_summary("isolated")

        assert summary["id"] == "isolated"
        assert summary["attributes"]["label"] == "alone"
        assert summary["degree"] == 0
        assert summary["neighbors"] == []

    def test_get_node_summary_nonexistent_node(self):
        """Test getting summary of non-existent node."""
        graph = nx.Graph()
        graph.add_node(1)
        node_ops = NodeOperations(graph)

        with pytest.raises(ValueError) as exc:
            node_ops.get_node_summary(99)
        assert "Node 99 not found" in str(exc.value)

    def test_get_node_summary_directed_graph(self):
        """Test node summary in directed graph."""
        graph = nx.DiGraph()
        graph.add_node("A")
        graph.add_edges_from([("A", "B"), ("A", "C"), ("D", "A")])
        node_ops = NodeOperations(graph)

        summary = node_ops.get_node_summary("A")

        assert summary["id"] == "A"
        assert summary["degree"] == 3  # Total degree (in + out)
        # In DiGraph, neighbors are successors only
        assert set(summary["neighbors"]) == {"B", "C"}

    def test_mixed_operations(self):
        """Test combining node and edge operations."""
        graph = nx.Graph()
        node_ops = NodeOperations(graph)
        edge_ops = EdgeOperations(graph)

        # Add nodes
        node_ops.add_node_with_validation(1, type="source")
        node_ops.add_node_with_validation(2, type="target")

        # Add edge
        edge_ops.add_edge_with_validation(1, 2, weight=1.5)

        # Check summaries
        node_summary = node_ops.get_node_summary(1)
        assert node_summary["degree"] == 1
        assert 2 in node_summary["neighbors"]

        edge_summary = edge_ops.get_edge_summary(1, 2)
        assert edge_summary["weight"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
