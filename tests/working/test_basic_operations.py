"""
Tests for basic graph operations using the minimal server.

These tests actually run and verify that the core functionality works.
"""

import pytest

from networkx_mcp.errors import GraphNotFoundError, InvalidEdgeError, MCPError
from networkx_mcp.core.basic_operations import (
    add_edges,
    add_nodes,
    betweenness_centrality,
    community_detection,
    connected_components,
    create_graph,
    degree_centrality,
    export_json,
    get_graph_info,
    import_csv,
    pagerank,
    shortest_path,
    visualize_graph,
)
from networkx_mcp.graph_cache import graphs


class TestGraphOperations:
    """Test basic graph operations."""

    def test_create_graph_undirected(self):
        """Test creating an undirected graph."""
        result = create_graph("test_undirected", directed=False, graphs=graphs)

        assert result["created"]
        assert result["graph_id"] == "test_undirected"
        assert not result["metadata"]["attributes"]["directed"]
        assert "test_undirected" in graphs
        assert not graphs["test_undirected"].is_directed()

    def test_create_graph_directed(self):
        """Test creating a directed graph."""
        result = create_graph("test_directed", directed=True, graphs=graphs)

        assert result["created"]
        assert result["graph_id"] == "test_directed"
        assert result["metadata"]["attributes"]["directed"]
        assert "test_directed" in graphs
        assert graphs["test_directed"].is_directed()

    def test_add_nodes(self):
        """Test adding nodes to a graph."""
        create_graph("test_nodes", directed=False, graphs=graphs)
        result = add_nodes("test_nodes", [1, 2, 3, 4, 5], graphs=graphs)

        assert result["success"]
        assert result["nodes_added"] == 5
        assert result["total"] == 5

        graph = graphs["test_nodes"]
        assert set(graph.nodes()) == {1, 2, 3, 4, 5}

    def test_add_edges(self):
        """Test adding edges to a graph."""
        create_graph("test_edges", directed=False, graphs=graphs)
        add_nodes("test_edges", [1, 2, 3, 4], graphs=graphs)
        result = add_edges("test_edges", [[1, 2], [2, 3], [3, 4]], graphs=graphs)

        assert result["success"]
        assert result["edges_added"] == 3
        assert result["total"] == 3

        graph = graphs["test_edges"]
        expected_edges = {(1, 2), (2, 3), (3, 4)}
        actual_edges = set(graph.edges())
        assert actual_edges == expected_edges

    def test_get_graph_info(self):
        """Test getting graph information."""
        create_graph("test_info", directed=False, graphs=graphs)
        add_nodes("test_info", [1, 2, 3], graphs=graphs)
        add_edges("test_info", [[1, 2], [2, 3]], graphs=graphs)

        result = get_graph_info("test_info", graphs=graphs)

        assert result["num_nodes"] == 3
        assert result["num_edges"] == 2
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2
        assert not result["directed"]

    def test_shortest_path(self):
        """Test shortest path calculation."""
        create_graph("test_path", directed=False, graphs=graphs)
        add_nodes("test_path", [1, 2, 3, 4, 5], graphs=graphs)
        add_edges("test_path", [[1, 2], [2, 3], [3, 4], [4, 5]], graphs=graphs)

        result = shortest_path("test_path", 1, 5, graphs=graphs)

        assert result["path"] == [1, 2, 3, 4, 5]
        assert result["length"] == 4


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_add_nodes_nonexistent_graph(self):
        """Test adding nodes to a non-existent graph."""
        with pytest.raises(GraphNotFoundError):
            add_nodes("nonexistent", [1, 2, 3], graphs=graphs)

    def test_add_edges_malformed_raises(self):
        """Test that malformed edges raise InvalidEdgeError instead of silently dropping."""
        create_graph("test_malformed", directed=False, graphs=graphs)
        add_nodes("test_malformed", [1, 2], graphs=graphs)
        with pytest.raises(InvalidEdgeError):
            add_edges("test_malformed", [[1]], graphs=graphs)  # Too few elements

    def test_add_edges_nonexistent_graph(self):
        """Test adding edges to a non-existent graph."""
        with pytest.raises(GraphNotFoundError):
            add_edges("nonexistent", [[1, 2]], graphs=graphs)

    def test_get_info_nonexistent_graph(self):
        """Test getting info for a non-existent graph."""
        result = get_graph_info("nonexistent", graphs=graphs)
        assert not result["success"]
        assert "not found" in result["error"]

    def test_shortest_path_nonexistent_graph(self):
        """Test shortest path on a non-existent graph."""
        result = shortest_path("nonexistent", 1, 2, graphs=graphs)
        assert not result["success"]
        assert "not found" in result["error"]

    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists."""
        create_graph("test_no_path", directed=False, graphs=graphs)
        add_nodes("test_no_path", [1, 2, 3, 4], graphs=graphs)
        add_edges(
            "test_no_path", [[1, 2]], graphs=graphs
        )  # Only connect 1-2, leave 3-4 isolated

        result = shortest_path("test_no_path", 1, 3, graphs=graphs)
        assert not result["success"]
        assert "No path found" in result["error"]

    def test_import_csv_rejects_oversized_data(self):
        """Test that CSV import rejects data exceeding 10MB limit."""
        from networkx_mcp.errors import ResourceLimitExceededError

        oversized = "a,b\n" * 5_000_001  # > 10MB
        with pytest.raises(ResourceLimitExceededError):
            import_csv("csv_g", oversized, graphs=graphs)


class TestComplexScenarios:
    """Test more complex scenarios."""

    def test_multiple_graphs(self):
        """Test managing multiple graphs simultaneously."""
        # Create multiple graphs
        create_graph("graph1", directed=False, graphs=graphs)
        create_graph("graph2", directed=True, graphs=graphs)
        create_graph("graph3", directed=False, graphs=graphs)

        # Add different data to each
        add_nodes("graph1", [1, 2, 3], graphs=graphs)
        add_nodes("graph2", ["a", "b", "c"], graphs=graphs)
        add_nodes("graph3", [10, 20, 30], graphs=graphs)

        # Verify they're independent
        info1 = get_graph_info("graph1", graphs=graphs)
        info2 = get_graph_info("graph2", graphs=graphs)
        info3 = get_graph_info("graph3", graphs=graphs)

        assert info1["num_nodes"] == 3
        assert info2["num_nodes"] == 3
        assert info3["num_nodes"] == 3

        assert not info1["is_directed"]
        assert info2["is_directed"]
        assert not info3["is_directed"]

    @pytest.mark.timeout(30)
    def test_large_graph(self):
        """Test with a reasonably large graph."""
        create_graph("large_graph", directed=False, graphs=graphs)

        # Create a 100-node graph
        nodes = list(range(100))
        add_nodes("large_graph", nodes, graphs=graphs)

        # Create a path graph
        edges = [[i, i + 1] for i in range(99)]
        add_edges("large_graph", edges, graphs=graphs)

        info = get_graph_info("large_graph", graphs=graphs)
        assert info["num_nodes"] == 100
        assert info["num_edges"] == 99

        # Test shortest path from start to end
        result = shortest_path("large_graph", 0, 99, graphs=graphs)
        assert result["length"] == 99
        assert len(result["path"]) == 100


# ===========================================================================
# None-graphs parametrized tests
# ===========================================================================

# All 13 compatibility functions from basic_operations.py that accept graphs=
_FUNCS_REQUIRING_GRAPH_NAME = [
    ("create_graph", create_graph, {"name": "g", "directed": False}),
    ("add_nodes", add_nodes, {"graph_name": "missing", "nodes": [1]}),
    ("add_edges", add_edges, {"graph_name": "missing", "edges": [[1, 2]]}),
    ("get_graph_info", get_graph_info, {"graph_name": "missing"}),
    (
        "shortest_path",
        shortest_path,
        {"graph_name": "missing", "source": 1, "target": 2},
    ),
    ("degree_centrality", degree_centrality, {"graph_name": "missing"}),
    ("betweenness_centrality", betweenness_centrality, {"graph_name": "missing"}),
    ("connected_components", connected_components, {"graph_name": "missing"}),
    ("pagerank", pagerank, {"graph_name": "missing"}),
    ("visualize_graph", visualize_graph, {"graph_name": "missing"}),
    ("import_csv", import_csv, {"graph_name": "g", "csv_data": "a,b\n1,2"}),
    ("export_json", export_json, {"graph_name": "missing"}),
    ("community_detection", community_detection, {"graph_name": "missing"}),
]


@pytest.mark.parametrize(
    "name,func,kwargs",
    _FUNCS_REQUIRING_GRAPH_NAME,
    ids=[t[0] for t in _FUNCS_REQUIRING_GRAPH_NAME],
)
def test_functions_with_none_graphs(name, func, kwargs):
    """Passing graphs=None (the default) should not crash — it either
    succeeds (create_graph, import_csv) or raises/returns an error dict."""
    try:
        result = func(**kwargs, graphs=None)
        # Functions that create graphs (create_graph, import_csv) should succeed
        # Functions that look up a graph should return error dict or raise
        if isinstance(result, dict) and "success" in result:
            # get_graph_info and shortest_path return {"success": False, ...}
            assert not result["success"] or name in ("create_graph", "import_csv")
    except (ValueError, MCPError):
        # add_nodes, add_edges, etc. raise ValueError;
        # export_json raises GraphNotFoundError (MCPError subclass)
        pass


# ===========================================================================
# Kamada-Kawai layout
# ===========================================================================


class TestVisualizeKamadaKwai:
    def test_visualize_kamada_kawai_layout(self):
        """Kamada-Kawai layout option produces a valid result."""
        create_graph("kk_graph", directed=False, graphs=graphs)
        add_nodes("kk_graph", [1, 2, 3], graphs=graphs)
        add_edges("kk_graph", [[1, 2], [2, 3]], graphs=graphs)

        result = visualize_graph("kk_graph", layout="kamada_kawai", graphs=graphs)
        assert result["layout"] == "kamada_kwai" or result["layout"] == "kamada_kawai"
        assert result["format"] == "png"
        assert result["image"].startswith("data:image/png;base64,")


# ===========================================================================
# Shortest path — node not found
# ===========================================================================


class TestShortestPathNodeNotFound:
    def test_shortest_path_node_not_found(self):
        """Requesting a path with a non-existent node returns an error."""
        create_graph("sp_missing", directed=False, graphs=graphs)
        add_nodes("sp_missing", [1, 2, 3], graphs=graphs)
        add_edges("sp_missing", [[1, 2], [2, 3]], graphs=graphs)

        result = shortest_path("sp_missing", 1, 999, graphs=graphs)
        assert result["success"] is False
        assert "not found" in result["error"].lower() or "No path" in result["error"]
