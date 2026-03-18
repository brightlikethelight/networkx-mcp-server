"""
Tests for new graph operations added in Phase 4.

These tests verify the enhanced functionality works correctly.
"""

import pytest

from networkx_mcp.core.basic_operations import (
    add_edges,
    add_nodes,
    betweenness_centrality,
    community_detection,
    connected_components,
    create_graph,
    degree_centrality,
    export_json,
    import_csv,
    pagerank,
    visualize_graph,
)
from networkx_mcp.graph_cache import graphs


class TestCentralityOperations:
    """Test centrality algorithms."""

    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()

    def test_degree_centrality(self):
        """Test degree centrality calculation."""
        create_graph("test_degree", directed=False, graphs=graphs)
        add_nodes("test_degree", ["A", "B", "C", "D", "E"], graphs=graphs)
        add_edges(
            "test_degree",
            [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["D", "E"]],
            graphs=graphs,
        )

        result = degree_centrality("test_degree", graphs=graphs)

        assert "centrality" in result
        assert "most_central" in result
        # Node A has 3 connections out of 4 possible (n-1)
        assert result["most_central"][0] == "A"
        assert result["most_central"][1] == 0.75  # 3/4

    def test_betweenness_centrality(self):
        """Test betweenness centrality calculation."""
        create_graph("test_between", directed=False, graphs=graphs)
        # Create a graph where B is a bridge between two clusters
        add_nodes("test_between", ["A", "B", "C", "D", "E"], graphs=graphs)
        add_edges(
            "test_between",
            [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["E", "C"]],
            graphs=graphs,
        )

        result = betweenness_centrality("test_between", graphs=graphs)

        assert "centrality" in result
        assert "most_central" in result
        # C should have high betweenness as it's on many shortest paths
        assert result["most_central"][0] == "C"

    def test_pagerank(self):
        """Test PageRank calculation."""
        create_graph("test_pagerank", directed=True, graphs=graphs)
        add_nodes("test_pagerank", ["A", "B", "C", "D"], graphs=graphs)
        # Create a directed graph where many point to A
        add_edges(
            "test_pagerank",
            [["B", "A"], ["C", "A"], ["D", "A"], ["A", "B"]],
            graphs=graphs,
        )

        result = pagerank("test_pagerank", graphs=graphs)

        assert "pagerank" in result
        assert "highest_rank" in result
        # A should have highest PageRank as it receives most links
        assert result["highest_rank"][0] == "A"


class TestGraphStructureOperations:
    """Test graph structure analysis."""

    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()

    def test_connected_components_undirected(self):
        """Test connected components in undirected graph."""
        create_graph("test_components", directed=False, graphs=graphs)
        # Create two separate components
        add_nodes("test_components", [1, 2, 3, 4, 5, 6], graphs=graphs)
        add_edges("test_components", [[1, 2], [2, 3], [4, 5], [5, 6]], graphs=graphs)

        result = connected_components("test_components", graphs=graphs)

        assert result["num_components"] == 2
        assert result["component_sizes"] == [3, 3]
        assert len(result["largest_component"]) == 3

    def test_connected_components_directed(self):
        """Test weakly connected components in directed graph."""
        create_graph("test_directed_comp", directed=True, graphs=graphs)
        add_nodes("test_directed_comp", ["A", "B", "C", "D"], graphs=graphs)
        add_edges("test_directed_comp", [["A", "B"], ["C", "D"]], graphs=graphs)

        result = connected_components("test_directed_comp", graphs=graphs)

        assert result["num_components"] == 2
        assert result["component_sizes"] == [2, 2]

    def test_community_detection(self):
        """Test community detection."""
        create_graph("test_communities", directed=False, graphs=graphs)
        # Create two clear communities
        add_nodes("test_communities", list(range(1, 9)), graphs=graphs)
        # Community 1: 1-4 fully connected
        add_edges(
            "test_communities",
            [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            graphs=graphs,
        )
        # Community 2: 5-8 fully connected
        add_edges(
            "test_communities",
            [[5, 6], [5, 7], [5, 8], [6, 7], [6, 8], [7, 8]],
            graphs=graphs,
        )
        # Weak link between communities
        add_edges("test_communities", [[4, 5]], graphs=graphs)

        result = community_detection("test_communities", graphs=graphs)

        assert result["num_communities"] >= 2
        assert "community_sizes" in result
        assert "largest_community" in result
        assert "node_community_map" in result


class TestVisualization:
    """Test graph visualization."""

    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()

    def test_visualize_graph_spring_layout(self):
        """Test graph visualization with spring layout."""
        create_graph("test_viz", directed=False, graphs=graphs)
        add_nodes("test_viz", ["A", "B", "C", "D"], graphs=graphs)
        add_edges(
            "test_viz", [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]], graphs=graphs
        )

        result = visualize_graph("test_viz", layout="spring", graphs=graphs)

        assert "image" in result
        assert result["image"].startswith("data:image/png;base64,")
        assert result["format"] == "png"
        assert result["layout"] == "spring"

    def test_visualize_graph_circular_layout(self):
        """Test graph visualization with circular layout."""
        create_graph("test_viz_circular", directed=True, graphs=graphs)
        add_nodes("test_viz_circular", [1, 2, 3, 4, 5], graphs=graphs)
        add_edges(
            "test_viz_circular",
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]],
            graphs=graphs,
        )

        result = visualize_graph("test_viz_circular", layout="circular", graphs=graphs)

        assert "image" in result
        assert result["image"].startswith("data:image/png;base64,")
        assert result["layout"] == "circular"


class TestImportExport:
    """Test import/export functionality."""

    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()

    def test_import_csv_undirected(self):
        """Test importing graph from CSV."""
        csv_data = """1,2
2,3
3,4
4,1"""

        result = import_csv("test_import", csv_data, directed=False, graphs=graphs)

        assert result["imported"] == "test_import"
        assert result["type"] == "undirected"
        assert result["nodes"] == 4
        assert result["edges"] == 4

    def test_import_csv_directed(self):
        """Test importing directed graph from CSV."""
        csv_data = """A,B
B,C
C,A"""

        result = import_csv("test_import_dir", csv_data, directed=True, graphs=graphs)

        assert result["imported"] == "test_import_dir"
        assert result["type"] == "directed"
        assert result["nodes"] == 3
        assert result["edges"] == 3

    def test_export_json(self):
        """Test exporting graph as JSON."""
        create_graph("test_export", directed=False, graphs=graphs)
        add_nodes("test_export", ["X", "Y", "Z"], graphs=graphs)
        add_edges("test_export", [["X", "Y"], ["Y", "Z"]], graphs=graphs)

        result = export_json("test_export", graphs=graphs)

        assert "graph_data" in result
        assert result["format"] == "node-link"
        assert result["nodes"] == 3
        assert result["edges"] == 2

        # Verify the structure
        data = result["graph_data"]
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 3
        assert len(data["links"]) == 2


class TestErrorHandlingNew:
    """Test error handling for new operations."""

    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()

    def test_centrality_nonexistent_graph(self):
        """Test centrality operations on non-existent graph."""
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            degree_centrality("nonexistent", graphs=graphs)

        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            betweenness_centrality("nonexistent", graphs=graphs)

        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            pagerank("nonexistent", graphs=graphs)

    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        create_graph("empty", directed=False, graphs=graphs)

        # These should work but return empty results
        result = degree_centrality("empty", graphs=graphs)
        assert result["most_central"] is None

        result = connected_components("empty", graphs=graphs)
        assert result["num_components"] == 0

        result = community_detection("empty", graphs=graphs)
        assert result["num_communities"] == 0
