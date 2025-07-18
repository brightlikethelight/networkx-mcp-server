"""No fake tests - each tool must work with real MCP protocol."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from networkx_mcp.server import (
    add_edges,
    add_nodes,
    centrality_measures,
    create_graph,
    delete_graph,
    get_graph_info,
    graph_manager,  # Access to manager for verification
    shortest_path,
)


class TestActualTools:
    """Test each tool actually works - no mocking, no faking."""

    def setup_method(self):
        """Clean state before each test."""
        # Clear any existing graphs
        for graph_id in list(graph_manager.graphs.keys()):
            graph_manager.delete_graph(graph_id)

    def test_create_graph_actually_works(self):
        """Test create_graph tool creates a real graph."""
        # Create undirected graph
        result = create_graph("test_graph", directed=False)

        # Verify result
        assert result is not None
        assert "error" not in result
        assert result.get("created") is True
        assert result.get("graph_id") == "test_graph"
        assert result.get("metadata", {}).get("attributes", {}).get("directed") is False

        # Verify graph actually exists in manager
        assert "test_graph" in graph_manager.graphs
        graph = graph_manager.get_graph("test_graph")
        assert graph is not None
        assert not graph.is_directed()

        # Create directed graph
        result2 = create_graph("directed_graph", directed=True)
        assert result2.get("created") is True

        # Verify directed graph
        directed = graph_manager.get_graph("directed_graph")
        assert directed.is_directed()

    def test_add_nodes_actually_works(self):
        """Test add_nodes tool adds real nodes."""
        # Create graph first
        create_graph("node_test")

        # Add nodes
        nodes = ["A", "B", "C", "D"]
        result = add_nodes("node_test", nodes)

        # Verify result
        assert result is not None
        assert "error" not in result
        assert result.get("success") is True
        assert result.get("nodes_added") == 4

        # Verify nodes actually exist
        graph = graph_manager.get_graph("node_test")
        assert set(graph.nodes()) == set(nodes)

        # Test duplicate nodes don't break things
        result2 = add_nodes("node_test", ["A", "E"])
        assert result2.get("nodes_added") == 1  # Only E is new
        assert "E" in graph.nodes()

    def test_add_edges_actually_works(self):
        """Test add_edges tool adds real edges."""
        # Create graph and add nodes
        create_graph("edge_test")
        add_nodes("edge_test", ["A", "B", "C", "D"])

        # Add edges
        edges = [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]]
        result = add_edges("edge_test", edges)

        # Verify result
        assert result is not None
        assert "error" not in result
        assert result.get("success") is True
        assert result.get("edges_added") == 4

        # Verify edges actually exist
        graph = graph_manager.get_graph("edge_test")
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert graph.has_edge("C", "D")
        assert graph.has_edge("D", "A")
        assert graph.number_of_edges() == 4

    def test_get_graph_info_actually_works(self):
        """Test get_graph_info returns real information."""
        # Create graph with nodes and edges
        create_graph("info_test", directed=True)
        add_nodes("info_test", ["X", "Y", "Z"])
        add_edges("info_test", [["X", "Y"], ["Y", "Z"]])

        # Get info
        result = get_graph_info("info_test")

        # Verify result
        assert result is not None
        assert "error" not in result
        assert result.get("graph_id") == "info_test"
        assert result.get("num_nodes") == 3
        assert result.get("num_edges") == 2
        assert result.get("is_directed") is True
        assert set(result.get("nodes", [])) == {"X", "Y", "Z"}
        assert result.get("edges") == [["X", "Y"], ["Y", "Z"]]

        # Test non-existent graph
        result2 = get_graph_info("does_not_exist")
        assert result2.get("success") is False
        assert "error" in result2

    def test_shortest_path_actually_works(self):
        """Test shortest_path finds real paths."""
        # Create a simple path graph
        create_graph("path_test")
        add_nodes("path_test", ["A", "B", "C", "D", "E"])
        add_edges(
            "path_test",
            [
                ["A", "B"],
                ["B", "C"],
                ["C", "D"],
                ["D", "E"],
                ["A", "E"],  # Shortcut
            ],
        )

        # Find shortest path (should use shortcut)
        result = shortest_path("path_test", "A", "E")

        # Verify result
        assert result is not None
        assert result.get("success") is True
        assert result.get("path") == ["A", "E"]  # Direct path

        # Find longer path
        result2 = shortest_path("path_test", "A", "D")
        assert result2.get("success") is True
        assert result2.get("path") == ["A", "B", "C", "D"]

        # Test no path exists
        add_nodes("path_test", ["Z"])  # Isolated node
        result3 = shortest_path("path_test", "A", "Z")
        assert result3.get("success") is False
        assert "No path" in result3.get("error", "")

    def test_centrality_measures_actually_works(self):
        """Test centrality_measures calculates real metrics."""
        # Create a star graph (node C is central)
        create_graph("centrality_test")
        add_nodes("centrality_test", ["A", "B", "C", "D", "E"])
        add_edges("centrality_test", [["C", "A"], ["C", "B"], ["C", "D"], ["C", "E"]])

        # Calculate all centrality measures
        result = centrality_measures(
            "centrality_test", ["degree", "betweenness", "closeness", "eigenvector"]
        )

        # Verify result structure
        assert result is not None
        assert result.get("success") is True
        assert "centrality" in result

        centrality = result["centrality"]

        # Verify degree centrality (C should be highest)
        assert "degree" in centrality
        degree = centrality["degree"]
        assert degree["C"] > degree["A"]
        assert degree["C"] > degree["B"]

        # Verify betweenness centrality (C should be highest)
        assert "betweenness" in centrality
        betweenness = centrality["betweenness"]
        assert betweenness["C"] > betweenness["A"]

        # Verify closeness centrality exists
        assert "closeness" in centrality

        # Verify eigenvector centrality exists
        assert "eigenvector" in centrality

        # Test with empty graph
        create_graph("empty_centrality")
        result2 = centrality_measures("empty_centrality", ["degree"])
        assert result2.get("success") is True
        assert result2["centrality"]["degree"] == {}

    def test_delete_graph_actually_works(self):
        """Test delete_graph removes real graphs."""
        # Create and verify graph exists
        create_graph("delete_test")
        add_nodes("delete_test", ["A", "B"])
        assert "delete_test" in graph_manager.graphs

        # Delete graph
        result = delete_graph("delete_test")

        # Verify result
        assert result is not None
        assert result.get("success") is True
        assert result.get("graph_id") == "delete_test"
        assert result.get("deleted") is True

        # Verify graph is actually gone
        assert "delete_test" not in graph_manager.graphs

        # Try to delete non-existent graph
        result2 = delete_graph("does_not_exist")
        assert result2.get("success") is False
        assert "not found" in result2.get("error", "").lower()

    def test_integration_workflow(self):
        """Test a complete workflow using multiple tools."""
        # 1. Create a social network graph
        create_graph("social_network")

        # 2. Add people
        people = ["Alice", "Bob", "Charlie", "David", "Eve"]
        add_nodes("social_network", people)

        # 3. Add friendships
        friendships = [
            ["Alice", "Bob"],
            ["Alice", "Charlie"],
            ["Bob", "Charlie"],
            ["Bob", "David"],
            ["Charlie", "David"],
            ["David", "Eve"],
        ]
        add_edges("social_network", friendships)

        # 4. Get graph info
        info = get_graph_info("social_network")
        assert info["num_nodes"] == 5
        assert info["num_edges"] == 6

        # 5. Find shortest path between Alice and Eve
        path = shortest_path("social_network", "Alice", "Eve")
        assert path["success"] is True
        assert len(path["path"]) >= 3  # At least 3 hops

        # 6. Calculate who's most central
        centrality = centrality_measures("social_network", ["degree", "betweenness"])

        # Bob, Charlie, and David should have higher centrality
        degree = centrality["centrality"]["degree"]
        assert degree["Bob"] >= degree["Alice"]
        assert degree["Bob"] >= degree["Eve"]

        # 7. Clean up
        delete_graph("social_network")
        assert "social_network" not in graph_manager.graphs


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
