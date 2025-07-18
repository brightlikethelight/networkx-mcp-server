"""
Tests for basic graph operations using the minimal server.

These tests actually run and verify that the core functionality works.
"""

import pytest

from networkx_mcp.server import (
    add_edges,
    add_nodes,
    create_graph,
    get_graph_info,
    graphs,
    shortest_path,
)


class TestGraphOperations:
    """Test basic graph operations."""

    def test_create_graph_undirected(self):
        """Test creating an undirected graph."""
        result = create_graph("test_undirected", directed=False)

        assert result["created"] == "test_undirected"
        assert result["type"] == "undirected"
        assert "test_undirected" in graphs
        assert not graphs["test_undirected"].is_directed()

    def test_create_graph_directed(self):
        """Test creating a directed graph."""
        result = create_graph("test_directed", directed=True)

        assert result["created"] == "test_directed"
        assert result["type"] == "directed"
        assert "test_directed" in graphs
        assert graphs["test_directed"].is_directed()

    def test_add_nodes(self):
        """Test adding nodes to a graph."""
        create_graph("test_nodes", directed=False)
        result = add_nodes("test_nodes", [1, 2, 3, 4, 5])

        assert result["added"] == 5
        assert result["total"] == 5

        graph = graphs["test_nodes"]
        assert set(graph.nodes()) == {1, 2, 3, 4, 5}

    def test_add_edges(self):
        """Test adding edges to a graph."""
        create_graph("test_edges", directed=False)
        add_nodes("test_edges", [1, 2, 3, 4])
        result = add_edges("test_edges", [[1, 2], [2, 3], [3, 4]])

        assert result["added"] == 3
        assert result["total"] == 3

        graph = graphs["test_edges"]
        expected_edges = {(1, 2), (2, 3), (3, 4)}
        actual_edges = set(graph.edges())
        assert actual_edges == expected_edges

    def test_get_graph_info(self):
        """Test getting graph information."""
        create_graph("test_info", directed=False)
        add_nodes("test_info", [1, 2, 3])
        add_edges("test_info", [[1, 2], [2, 3]])

        result = get_graph_info("test_info")

        assert result["nodes"] == 3
        assert result["edges"] == 2
        assert result["directed"] == False

    def test_shortest_path(self):
        """Test shortest path calculation."""
        create_graph("test_path", directed=False)
        add_nodes("test_path", [1, 2, 3, 4, 5])
        add_edges("test_path", [[1, 2], [2, 3], [3, 4], [4, 5]])

        result = shortest_path("test_path", 1, 5)

        assert result["path"] == [1, 2, 3, 4, 5]
        assert result["length"] == 4


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_add_nodes_nonexistent_graph(self):
        """Test adding nodes to a non-existent graph."""
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            add_nodes("nonexistent", [1, 2, 3])

    def test_add_edges_nonexistent_graph(self):
        """Test adding edges to a non-existent graph."""
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            add_edges("nonexistent", [[1, 2]])

    def test_get_info_nonexistent_graph(self):
        """Test getting info for a non-existent graph."""
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            get_graph_info("nonexistent")

    def test_shortest_path_nonexistent_graph(self):
        """Test shortest path on a non-existent graph."""
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            shortest_path("nonexistent", 1, 2)

    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists."""
        create_graph("test_no_path", directed=False)
        add_nodes("test_no_path", [1, 2, 3, 4])
        add_edges("test_no_path", [[1, 2]])  # Only connect 1-2, leave 3-4 isolated

        with pytest.raises(Exception):  # NetworkX raises NetworkXNoPath
            shortest_path("test_no_path", 1, 3)


class TestComplexScenarios:
    """Test more complex scenarios."""

    def test_multiple_graphs(self):
        """Test managing multiple graphs simultaneously."""
        # Create multiple graphs
        create_graph("graph1", directed=False)
        create_graph("graph2", directed=True)
        create_graph("graph3", directed=False)

        # Add different data to each
        add_nodes("graph1", [1, 2, 3])
        add_nodes("graph2", ["a", "b", "c"])
        add_nodes("graph3", [10, 20, 30])

        # Verify they're independent
        info1 = get_graph_info("graph1")
        info2 = get_graph_info("graph2")
        info3 = get_graph_info("graph3")

        assert info1["nodes"] == 3
        assert info2["nodes"] == 3
        assert info3["nodes"] == 3

        assert info1["directed"] == False
        assert info2["directed"] == True
        assert info3["directed"] == False

    def test_large_graph(self):
        """Test with a reasonably large graph."""
        create_graph("large_graph", directed=False)

        # Create a 100-node graph
        nodes = list(range(100))
        add_nodes("large_graph", nodes)

        # Create a path graph
        edges = [[i, i + 1] for i in range(99)]
        add_edges("large_graph", edges)

        info = get_graph_info("large_graph")
        assert info["nodes"] == 100
        assert info["edges"] == 99

        # Test shortest path from start to end
        result = shortest_path("large_graph", 0, 99)
        assert result["length"] == 99
        assert len(result["path"]) == 100
