"""Test core graph operations actually work."""

from networkx_mcp.core.graph_operations import GraphManager


def test_graph_lifecycle():
    """Test create, modify, delete graph."""
    gm = GraphManager()

    # Create graph
    result = gm.create_graph("test1", "DiGraph")
    assert result["created"] is True
    assert result["graph_id"] == "test1"

    # Add nodes
    result = gm.add_nodes_from("test1", ["A", "B", "C"])
    assert result["nodes_added"] == 3

    # Add edges
    result = gm.add_edges_from("test1", [("A", "B"), ("B", "C")])
    assert result["edges_added"] == 2

    # Get info
    info = gm.get_graph_info("test1")
    assert info["num_nodes"] == 3
    assert info["num_edges"] == 2

    # Delete
    result = gm.delete_graph("test1")
    assert result["deleted"] is True


def test_graph_manager_operations():
    """Test GraphManager operations with pytest."""
    gm = GraphManager()

    # Test graph creation
    result = gm.create_graph("pytest_graph", "Graph")
    assert result["created"] is True
    assert result["graph_id"] == "pytest_graph"

    # Test duplicate graph creation
    try:
        gm.create_graph("pytest_graph", "Graph")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already exists" in str(e)

    # Test node operations
    result = gm.add_nodes_from("pytest_graph", [1, 2, 3, 4, 5])
    assert result["nodes_added"] == 5

    # Test edge operations
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    result = gm.add_edges_from("pytest_graph", edges)
    assert result["edges_added"] == 5

    # Test graph info
    info = gm.get_graph_info("pytest_graph")
    assert info["num_nodes"] == 5
    assert info["num_edges"] == 5
    assert info["graph_type"] == "Graph"

    # Test listing graphs
    graphs = gm.list_graphs()
    graph_ids = [g["graph_id"] for g in graphs]
    assert "pytest_graph" in graph_ids

    # Clean up
    result = gm.delete_graph("pytest_graph")
    assert result["deleted"] is True


def test_directed_graph_operations():
    """Test directed graph specific operations."""
    gm = GraphManager()

    # Create directed graph
    result = gm.create_graph("directed_test", "DiGraph")
    assert result["created"] is True

    # Add nodes and edges
    gm.add_nodes_from("directed_test", ["A", "B", "C"])
    gm.add_edges_from("directed_test", [("A", "B"), ("B", "C"), ("C", "A")])

    # Get info should show directed properties
    info = gm.get_graph_info("directed_test")
    assert info["is_directed"] is True
    assert info["num_nodes"] == 3
    assert info["num_edges"] == 3

    # Clean up
    gm.delete_graph("directed_test")


def test_error_handling():
    """Test error handling in graph operations."""
    gm = GraphManager()

    # Test operations on non-existent graph
    try:
        gm.add_nodes_from("nonexistent", [1, 2, 3])
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "not found" in str(e)

    try:
        gm.add_edges_from("nonexistent", [(1, 2)])
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    try:
        gm.get_graph_info("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    try:
        gm.delete_graph("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_node_attributes():
    """Test node attribute operations."""
    gm = GraphManager()

    # Create graph
    gm.create_graph("attr_test", "Graph")

    # Add individual nodes with attributes
    result = gm.add_node("attr_test", 1, label="Node 1", weight=10)
    assert result["node_id"] == 1
    assert result["attributes"]["label"] == "Node 1"

    result = gm.add_node("attr_test", 2, label="Node 2", weight=20)
    assert result["node_id"] == 2

    # Get node attributes
    attrs = gm.get_node_attributes("attr_test", 1)
    assert attrs["label"] == "Node 1"
    assert attrs["weight"] == 10

    # Clean up
    gm.delete_graph("attr_test")


def test_edge_attributes():
    """Test edge attribute operations."""
    gm = GraphManager()

    # Create graph
    gm.create_graph("edge_attr_test", "Graph")
    gm.add_nodes_from("edge_attr_test", [1, 2, 3])

    # Add edge with attributes
    result = gm.add_edge("edge_attr_test", 1, 2, weight=5.0, label="Edge 1-2")
    assert result["edge"] == (1, 2)
    assert result["added"] is True
    assert result["attributes"]["weight"] == 5.0

    # Get edge attributes
    attrs = gm.get_edge_attributes("edge_attr_test", (1, 2))
    assert attrs["weight"] == 5.0
    assert attrs["label"] == "Edge 1-2"

    # Clean up
    gm.delete_graph("edge_attr_test")


def test_single_node_operations():
    """Test individual node operations."""
    gm = GraphManager()

    # Create graph
    gm.create_graph("single_node_test", "Graph")

    # Add single node
    result = gm.add_node("single_node_test", "node1", color="red")
    assert result["node_id"] == "node1"

    # Remove single node
    result = gm.remove_node("single_node_test", "node1")
    assert result["node_id"] == "node1"

    # Clean up
    gm.delete_graph("single_node_test")


def test_neighbors_and_subgraph():
    """Test neighbor finding and subgraph operations."""
    gm = GraphManager()

    # Create graph with some structure
    gm.create_graph("neighbor_test", "Graph")
    gm.add_nodes_from("neighbor_test", [1, 2, 3, 4, 5])
    gm.add_edges_from("neighbor_test", [(1, 2), (1, 3), (2, 4), (3, 5)])

    # Get neighbors of node 1
    neighbors = gm.get_neighbors("neighbor_test", 1)
    assert 2 in neighbors
    assert 3 in neighbors
    assert len(neighbors) == 2

    # Create subgraph (returns info, doesn't create new graph)
    result = gm.subgraph("neighbor_test", [1, 2, 3])
    assert result["num_nodes"] == 3
    assert result["num_edges"] == 2  # (1,2) and (1,3)
    assert set(result["nodes"]) == {1, 2, 3}

    # Clean up
    gm.delete_graph("neighbor_test")


# Run the basic test if executed directly
if __name__ == "__main__":
    print("🔍 Testing core graph operations...")
    test_graph_lifecycle()
    print("✅ Basic lifecycle test passed!")

    # Run more comprehensive tests
    print("\n🔍 Testing GraphManager operations...")
    test_graph_manager_operations()
    print("✅ GraphManager operations test passed!")

    print("\n🔍 Testing directed graph operations...")
    test_directed_graph_operations()
    print("✅ Directed graph test passed!")

    print("\n🔍 Testing error handling...")
    test_error_handling()
    print("✅ Error handling test passed!")

    print("\n🔍 Testing node attributes...")
    test_node_attributes()
    print("✅ Node attributes test passed!")

    print("\n🔍 Testing edge attributes...")
    test_edge_attributes()
    print("✅ Edge attributes test passed!")

    print("\n🔍 Testing neighbor and subgraph operations...")
    test_neighbors_and_subgraph()
    print("✅ Neighbor and subgraph test passed!")

    print("\n✅ All core operation tests passed!")
