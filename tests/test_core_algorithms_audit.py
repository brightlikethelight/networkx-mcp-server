"""Audit core algorithms functionality."""

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager


def test_basic_algorithms():
    """Test basic algorithm functionality."""
    gm = GraphManager()
    alg = GraphAlgorithms()

    # Create a test graph
    gm.create_graph("algo_test", "Graph")
    gm.add_nodes_from("algo_test", [1, 2, 3, 4, 5])
    gm.add_edges_from("algo_test", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])

    # Get the actual graph object
    graph = gm.get_graph("algo_test")

    # Test shortest path
    result = alg.shortest_path(graph, 1, 3)
    assert "path" in result
    assert "length" in result
    assert len(result["path"]) >= 2  # Should be at least source and target

    # Test connected components
    result = alg.connected_components(graph)
    assert "connected_components" in result
    assert result["num_components"] == 1  # Single connected component

    # Clean up
    gm.delete_graph("algo_test")


def test_directed_graph_algorithms():
    """Test algorithms on directed graphs."""
    gm = GraphManager()
    alg = GraphAlgorithms()

    # Create directed graph
    gm.create_graph("directed_algo", "DiGraph")
    gm.add_nodes_from("directed_algo", ["A", "B", "C", "D"])
    gm.add_edges_from("directed_algo", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])

    # Get the actual graph object
    graph = gm.get_graph("directed_algo")

    # Test connected components (should work on directed graphs too)
    result = alg.connected_components(graph)
    assert "weakly_connected_components" in result
    assert "strongly_connected_components" in result

    # Clean up
    gm.delete_graph("directed_algo")


def test_algorithm_error_handling():
    """Test algorithm error handling."""
    alg = GraphAlgorithms()
    gm = GraphManager()

    # Create empty graph for tests
    gm.create_graph("empty_test", "Graph")
    empty_graph = gm.get_graph("empty_test")

    # Test algorithms on empty graph (should work now after fix)
    result = alg.connected_components(empty_graph)
    assert "connected_components" in result
    assert result["num_components"] == 0
    assert result["is_connected"] is False
    assert result["largest_component_size"] == 0

    # Test shortest path on non-existent nodes
    gm.add_nodes_from("empty_test", [1, 2])
    graph_with_nodes = gm.get_graph("empty_test")

    try:
        alg.shortest_path(graph_with_nodes, 1, 3)  # Node 3 doesn't exist
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not in graph" in str(e)

    # Clean up
    gm.delete_graph("empty_test")


def test_clustering_algorithms():
    """Test clustering and community algorithms."""
    gm = GraphManager()
    alg = GraphAlgorithms()

    # Create graph with clear clustering structure
    gm.create_graph("cluster_test", "Graph")
    gm.add_nodes_from("cluster_test", [1, 2, 3, 4, 5, 6])
    gm.add_edges_from(
        "cluster_test",
        [
            (1, 2),
            (2, 3),
            (1, 3),  # Triangle 1
            (4, 5),
            (5, 6),
            (4, 6),  # Triangle 2
            (3, 4),  # Bridge
        ],
    )

    # Get the actual graph object
    graph = gm.get_graph("cluster_test")

    # Test clustering coefficient
    result = alg.clustering_coefficients(graph)
    assert "node_clustering" in result
    assert "average_clustering" in result
    assert len(result["node_clustering"]) == 6

    # Clean up
    gm.delete_graph("cluster_test")


def test_centrality_algorithms():
    """Test various centrality measures."""
    gm = GraphManager()
    alg = GraphAlgorithms()

    # Create star graph (one central node)
    gm.create_graph("star_test", "Graph")
    gm.add_nodes_from("star_test", [0, 1, 2, 3, 4])
    gm.add_edges_from("star_test", [(0, 1), (0, 2), (0, 3), (0, 4)])

    # Get the actual graph object
    graph = gm.get_graph("star_test")

    # Test centrality measures
    result = alg.centrality_measures(graph)
    assert "degree_centrality" in result
    assert "betweenness_centrality" in result
    assert "closeness_centrality" in result

    # Central node (0) should have high centrality values
    assert result["degree_centrality"][0] > 0.8

    # Clean up
    gm.delete_graph("star_test")


if __name__ == "__main__":
    print("🔍 Testing basic algorithms...")
    test_basic_algorithms()
    print("✅ Basic algorithms test passed!")

    print("\n🔍 Testing directed graph algorithms...")
    test_directed_graph_algorithms()
    print("✅ Directed graph algorithms test passed!")

    print("\n🔍 Testing algorithm error handling...")
    test_algorithm_error_handling()
    print("✅ Algorithm error handling test passed!")

    print("\n🔍 Testing clustering algorithms...")
    test_clustering_algorithms()
    print("✅ Clustering algorithms test passed!")

    print("\n🔍 Testing centrality algorithms...")
    test_centrality_algorithms()
    print("✅ Centrality algorithms test passed!")

    print("\n✅ All algorithm audit tests passed!")
