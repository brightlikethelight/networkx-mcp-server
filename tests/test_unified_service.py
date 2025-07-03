"""Test the UnifiedGraphService for consistent API."""

from networkx_mcp.services.unified_graph_service import UnifiedGraphService


def test_unified_service_consistent_api():
    """Test that the unified service provides consistent API across operations."""
    service = UnifiedGraphService()

    # Create graph
    result = service.create_graph("test_unified", "Graph")
    assert result["status"] == "success"
    assert result["graph_id"] == "test_unified"

    # Add nodes
    result = service.add_nodes("test_unified", [1, 2, 3, 4, 5])
    assert result["status"] == "success"
    assert result["nodes_added"] == 5

    # Add edges
    result = service.add_edges("test_unified", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    assert result["status"] == "success"
    assert result["edges_added"] == 5

    # Get graph info
    result = service.get_graph_info("test_unified")
    assert result["status"] == "success"
    assert result["num_nodes"] == 5
    assert result["num_edges"] == 5

    # Test algorithms with consistent API
    result = service.shortest_path("test_unified", 1, 3)
    assert result["status"] == "success"
    assert result["graph_id"] == "test_unified"
    assert "path" in result

    result = service.connected_components("test_unified")
    assert result["status"] == "success"
    assert result["graph_id"] == "test_unified"
    assert result["num_components"] == 1

    result = service.centrality_measures("test_unified")
    assert result["status"] == "success"
    assert result["graph_id"] == "test_unified"
    assert "degree_centrality" in result

    # Clean up
    result = service.delete_graph("test_unified")
    assert result["status"] == "success"


def test_unified_service_error_handling():
    """Test consistent error handling across the unified service."""
    service = UnifiedGraphService()

    # Test operations on non-existent graph
    result = service.add_nodes("nonexistent", [1, 2, 3])
    assert result["status"] == "error"
    assert "not found" in result["message"]

    result = service.shortest_path("nonexistent", 1, 2)
    assert result["status"] == "error"
    assert "graph_id" in result

    result = service.connected_components("nonexistent")
    assert result["status"] == "error"
    assert "graph_id" in result

    # Test duplicate graph creation
    service.create_graph("duplicate_test", "Graph")
    result = service.create_graph("duplicate_test", "Graph")
    assert result["status"] == "error"
    assert "already exists" in result["message"]

    # Clean up
    service.delete_graph("duplicate_test")


def test_unified_service_subgraph_improvement():
    """Test improved subgraph API that can create managed graphs."""
    service = UnifiedGraphService()

    # Create main graph
    service.create_graph("main_graph", "Graph")
    service.add_nodes("main_graph", [1, 2, 3, 4, 5, 6])
    service.add_edges("main_graph", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    # Create subgraph as new managed graph
    result = service.subgraph("main_graph", [1, 2, 3], new_graph_id="sub_graph")
    assert result["status"] == "success"
    assert result["subgraph_id"] == "sub_graph"
    assert result["num_nodes"] == 3
    assert result["num_edges"] == 2

    # Verify the subgraph exists as a managed graph
    result = service.get_graph_info("sub_graph")
    assert result["status"] == "success"
    assert result["num_nodes"] == 3

    # Test subgraph without creating new graph (original behavior)
    result = service.subgraph("main_graph", [4, 5, 6])
    assert result["status"] == "success"
    assert result["num_nodes"] == 3
    assert result["num_edges"] == 2
    assert "subgraph_id" not in result  # No new graph created

    # Clean up
    service.delete_graph("main_graph")
    service.delete_graph("sub_graph")


def test_unified_service_consistent_node_edge_operations():
    """Test that all node and edge operations have consistent APIs."""
    service = UnifiedGraphService()

    # Create graph
    service.create_graph("node_edge_test", "Graph")

    # Test single node operations
    result = service.add_node("node_edge_test", "A", label="Node A", weight=10)
    assert result["status"] == "success"
    assert result["node_id"] == "A"

    result = service.get_node_attributes("node_edge_test", "A")
    assert result["status"] == "success"
    assert result["attributes"]["label"] == "Node A"

    # Test bulk node operations
    result = service.add_nodes("node_edge_test", ["B", "C", "D"])
    assert result["status"] == "success"
    assert result["nodes_added"] == 3

    # Test edge operations
    result = service.add_edge("node_edge_test", "A", "B", weight=5.0)
    assert result["status"] == "success"
    assert result["edge"] == ("A", "B")

    result = service.get_edge_attributes("node_edge_test", "A", "B")
    assert result["status"] == "success"
    assert result["attributes"]["weight"] == 5.0

    # Test neighbors
    result = service.get_neighbors("node_edge_test", "A")
    assert result["status"] == "success"
    assert "B" in result["neighbors"]

    # Clean up
    service.delete_graph("node_edge_test")


def test_unified_service_empty_graph_algorithms():
    """Test that algorithms work correctly on empty graphs (bug fix verification)."""
    service = UnifiedGraphService()

    # Create empty graph
    service.create_graph("empty_test", "Graph")

    # Test algorithms on empty graph (should work after the fix)
    result = service.connected_components("empty_test")
    assert result["status"] == "success"
    assert result["num_components"] == 0
    assert result["is_connected"] is False

    result = service.centrality_measures("empty_test")
    assert result["status"] == "success"
    assert len(result["degree_centrality"]) == 0

    result = service.clustering_coefficients("empty_test")
    assert result["status"] == "success"
    assert result["average_clustering"] == 0.0

    # Clean up
    service.delete_graph("empty_test")


if __name__ == "__main__":
    print("🔍 Testing unified service consistent API...")
    test_unified_service_consistent_api()
    print("✅ Consistent API test passed!")

    print("\n🔍 Testing unified service error handling...")
    test_unified_service_error_handling()
    print("✅ Error handling test passed!")

    print("\n🔍 Testing improved subgraph API...")
    test_unified_service_subgraph_improvement()
    print("✅ Subgraph improvement test passed!")

    print("\n🔍 Testing consistent node/edge operations...")
    test_unified_service_consistent_node_edge_operations()
    print("✅ Consistent operations test passed!")

    print("\n🔍 Testing empty graph algorithms fix...")
    test_unified_service_empty_graph_algorithms()
    print("✅ Empty graph algorithms test passed!")

    print("\n✅ All unified service tests passed!")
