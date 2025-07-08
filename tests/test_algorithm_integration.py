#!/usr/bin/env python3
"""Comprehensive tests for all integrated algorithms from GraphAlgorithms."""

import pytest
import networkx as nx
from src.networkx_mcp.server import (
    create_graph, add_nodes, add_edges, 
    shortest_path, connected_components, centrality_measures,
    clustering_coefficients, minimum_spanning_tree, maximum_flow,
    graph_coloring, community_detection, cycles_detection,
    matching, graph_statistics, all_pairs_shortest_path
)


class TestAlgorithmIntegration:
    """Test all algorithms integrated from GraphAlgorithms."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test."""
        # Setup is done per test
        yield
        # Cleanup - delete any test graphs
        from src.networkx_mcp.server import delete_graph, list_graphs
        graphs = list_graphs()
        for graph in graphs.get("graphs", []):
            if graph["name"].startswith("test_"):
                delete_graph(graph["name"])
    
    def test_shortest_path_algorithms(self):
        """Test shortest path with different configurations."""
        # Create weighted graph
        create_graph("test_weighted", "undirected")
        add_nodes("test_weighted", ["A", "B", "C", "D", "E"])
        add_edges("test_weighted", [
            ["A", "B", {"weight": 1}],
            ["A", "C", {"weight": 4}],
            ["B", "C", {"weight": 2}],
            ["B", "D", {"weight": 5}],
            ["C", "D", {"weight": 1}],
            ["D", "E", {"weight": 3}]
        ])
        
        # Test unweighted shortest path
        result = shortest_path("test_weighted", "A", "E")
        assert result["success"] is True
        assert len(result["path"]) > 0
        
        # Test weighted shortest path
        result = shortest_path("test_weighted", "A", "E", weight="weight")
        assert result["success"] is True
        assert result["weighted"] is True
        assert result["length"] == 7  # A->B->C->D->E = 1+2+1+3
        
        print("✅ Shortest path algorithms work correctly")
    
    def test_connected_components(self):
        """Test connected components for various graph types."""
        # Test disconnected undirected graph
        create_graph("test_disconnected", "undirected")
        add_nodes("test_disconnected", [1, 2, 3, 4, 5, 6])
        add_edges("test_disconnected", [[1, 2], [2, 3], [4, 5]])
        
        result = connected_components("test_disconnected")
        assert result["num_components"] == 3
        assert result["is_connected"] is False
        assert result["largest_component_size"] == 3
        
        # Test directed graph
        create_graph("test_directed", "directed")
        add_nodes("test_directed", ["A", "B", "C", "D"])
        add_edges("test_directed", [["A", "B"], ["B", "C"], ["C", "A"], ["D", "D"]])
        
        result = connected_components("test_directed")
        assert "weakly_connected_components" in result
        assert "strongly_connected_components" in result
        assert result["num_weakly_connected"] == 2
        
        print("✅ Connected components analysis works correctly")
    
    def test_centrality_measures(self):
        """Test various centrality measures."""
        # Create star graph for clear centrality
        create_graph("test_star", "undirected")
        add_nodes("test_star", ["center", "n1", "n2", "n3", "n4"])
        add_edges("test_star", [
            ["center", "n1"], ["center", "n2"], 
            ["center", "n3"], ["center", "n4"]
        ])
        
        result = centrality_measures("test_star", ["degree", "betweenness", "closeness"])
        
        # Check degree centrality
        assert "degree_centrality" in result
        assert result["degree_centrality"]["center"] == 1.0  # Normalized to 1
        
        # Check betweenness centrality
        assert "betweenness_centrality" in result
        assert result["betweenness_centrality"]["center"] == 1.0
        
        # Check closeness centrality
        assert "closeness_centrality" in result
        assert result["closeness_centrality"]["center"] == 1.0
        
        print("✅ Centrality measures calculate correctly")
    
    def test_clustering_coefficients(self):
        """Test clustering coefficient calculations."""
        # Create triangle graph (high clustering)
        create_graph("test_triangle", "undirected")
        add_nodes("test_triangle", ["A", "B", "C", "D"])
        add_edges("test_triangle", [
            ["A", "B"], ["B", "C"], ["C", "A"],  # Triangle
            ["C", "D"]  # Extension
        ])
        
        result = clustering_coefficients("test_triangle")
        assert "node_clustering" in result
        assert "average_clustering" in result
        assert "transitivity" in result
        
        # Node A, B, C should have clustering coefficient 1.0 (in triangle)
        assert result["node_clustering"]["A"] == 1.0
        assert result["node_clustering"]["B"] == 1.0
        assert result["node_clustering"]["C"] == 0.33333333333333331  # 1 triangle out of 3 possible
        
        print("✅ Clustering coefficients calculate correctly")
    
    def test_minimum_spanning_tree(self):
        """Test MST algorithms."""
        # Create weighted graph
        create_graph("test_mst", "undirected")
        add_nodes("test_mst", ["A", "B", "C", "D"])
        add_edges("test_mst", [
            ["A", "B", {"weight": 1}],
            ["B", "C", {"weight": 2}],
            ["C", "D", {"weight": 1}],
            ["D", "A", {"weight": 3}],
            ["A", "C", {"weight": 4}]
        ])
        
        # Test Kruskal's algorithm
        result = minimum_spanning_tree("test_mst", weight="weight", algorithm="kruskal")
        assert result["total_weight"] == 4  # A-B(1) + C-D(1) + B-C(2)
        assert result["num_edges"] == 3
        assert result["algorithm"] == "kruskal"
        
        # Test Prim's algorithm
        result = minimum_spanning_tree("test_mst", weight="weight", algorithm="prim")
        assert result["total_weight"] == 4
        assert result["num_edges"] == 3
        
        print("✅ Minimum spanning tree algorithms work correctly")
    
    def test_maximum_flow(self):
        """Test maximum flow calculation."""
        # Create flow network
        create_graph("test_flow", "directed")
        add_nodes("test_flow", ["s", "a", "b", "t"])
        add_edges("test_flow", [
            ["s", "a", {"capacity": 10}],
            ["s", "b", {"capacity": 10}],
            ["a", "b", {"capacity": 2}],
            ["a", "t", {"capacity": 4}],
            ["b", "t", {"capacity": 10}]
        ])
        
        result = maximum_flow("test_flow", "s", "t", capacity="capacity")
        assert result["flow_value"] == 14  # 4 through s->a->t, 10 through s->b->t
        assert result["source"] == "s"
        assert result["sink"] == "t"
        assert "flow_dict" in result
        
        print("✅ Maximum flow calculation works correctly")
    
    def test_graph_coloring(self):
        """Test graph coloring algorithm."""
        # Create graph that needs 3 colors (triangle + extension)
        create_graph("test_coloring", "undirected")
        add_nodes("test_coloring", ["A", "B", "C", "D"])
        add_edges("test_coloring", [
            ["A", "B"], ["B", "C"], ["C", "A"],  # Triangle needs 3 colors
            ["A", "D"]
        ])
        
        result = graph_coloring("test_coloring", strategy="largest_first")
        assert "coloring" in result
        assert "num_colors" in result
        assert result["num_colors"] >= 3  # Triangle needs at least 3 colors
        
        # Check that adjacent nodes have different colors
        coloring = result["coloring"]
        assert coloring["A"] != coloring["B"]
        assert coloring["B"] != coloring["C"]
        assert coloring["C"] != coloring["A"]
        assert coloring["A"] != coloring["D"]
        
        print("✅ Graph coloring works correctly")
    
    def test_community_detection(self):
        """Test community detection algorithms."""
        # Skip if community algorithms not available
        try:
            # Create graph with clear communities
            create_graph("test_communities", "undirected")
            # Community 1
            add_nodes("test_communities", ["A", "B", "C"])
            add_edges("test_communities", [["A", "B"], ["B", "C"], ["C", "A"]])
            # Community 2
            add_nodes("test_communities", ["D", "E", "F"])
            add_edges("test_communities", [["D", "E"], ["E", "F"], ["F", "D"]])
            # Weak connection between communities
            add_edges("test_communities", [["C", "D"]])
            
            result = community_detection("test_communities", method="greedy_modularity")
            assert "communities" in result
            assert "num_communities" in result
            assert "modularity" in result
            assert result["num_communities"] >= 2
            
            print("✅ Community detection works correctly")
        except Exception as e:
            if "not available" in str(e):
                print("⚠️  Community detection not available in this NetworkX version")
            else:
                raise
    
    def test_cycles_detection(self):
        """Test cycle detection for different graph types."""
        # Test directed graph with cycle
        create_graph("test_cycle_directed", "directed")
        add_nodes("test_cycle_directed", ["A", "B", "C", "D"])
        add_edges("test_cycle_directed", [
            ["A", "B"], ["B", "C"], ["C", "A"],  # Cycle
            ["C", "D"]
        ])
        
        result = cycles_detection("test_cycle_directed")
        assert result["has_cycle"] is True
        assert result["is_dag"] is False
        assert "cycles" in result
        
        # Test undirected graph
        create_graph("test_cycle_undirected", "undirected")
        add_nodes("test_cycle_undirected", [1, 2, 3, 4])
        add_edges("test_cycle_undirected", [[1, 2], [2, 3], [3, 4], [4, 1]])
        
        result = cycles_detection("test_cycle_undirected")
        assert result["has_cycle"] is True
        assert "cycle_basis" in result
        assert len(result["cycle_basis"]) > 0
        
        print("✅ Cycle detection works correctly")
    
    def test_matching(self):
        """Test matching algorithms."""
        # Create bipartite-like graph
        create_graph("test_matching", "undirected")
        add_nodes("test_matching", ["A", "B", "C", "1", "2", "3"])
        add_edges("test_matching", [
            ["A", "1"], ["A", "2"],
            ["B", "2"], ["B", "3"],
            ["C", "3"]
        ])
        
        result = matching("test_matching", max_cardinality=True)
        assert "matching" in result
        assert "matching_size" in result
        assert "is_perfect" in result
        assert result["matching_size"] == 3  # Can match all nodes
        
        print("✅ Matching algorithms work correctly")
    
    def test_graph_statistics(self):
        """Test comprehensive graph statistics."""
        # Create graph with known properties
        create_graph("test_stats", "directed")
        add_nodes("test_stats", list(range(5)))
        add_edges("test_stats", [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # Cycle
            [0, 2], [1, 3]  # Additional edges
        ])
        
        result = graph_statistics("test_stats")
        assert result["num_nodes"] == 5
        assert result["num_edges"] == 7
        assert result["is_directed"] is True
        assert "density" in result
        assert "degree_stats" in result
        assert "in_degree_stats" in result
        assert "out_degree_stats" in result
        
        # Check degree statistics
        assert result["degree_stats"]["mean"] > 0
        assert result["in_degree_stats"]["min"] >= 0
        assert result["out_degree_stats"]["max"] > 0
        
        print("✅ Graph statistics calculate correctly")
    
    def test_all_pairs_shortest_path(self):
        """Test all pairs shortest path calculation."""
        # Create small graph (to avoid resource limits)
        create_graph("test_apsp", "undirected")
        add_nodes("test_apsp", ["A", "B", "C", "D"])
        add_edges("test_apsp", [
            ["A", "B", {"weight": 1}],
            ["B", "C", {"weight": 2}],
            ["C", "D", {"weight": 1}],
            ["A", "D", {"weight": 5}]
        ])
        
        result = all_pairs_shortest_path("test_apsp", weight="weight")
        assert "lengths" in result
        assert "paths" in result
        
        # Check specific path lengths
        lengths = result["lengths"]
        assert lengths["A"]["D"] == 4  # A->B->C->D is shorter than direct A->D
        assert lengths["B"]["D"] == 3  # B->C->D
        
        print("✅ All pairs shortest path works correctly")
    
    def test_algorithm_result_consistency(self):
        """Test that all algorithms return consistent result formats."""
        # Create a simple test graph
        create_graph("test_consistency", "undirected")
        add_nodes("test_consistency", [1, 2, 3, 4])
        add_edges("test_consistency", [[1, 2], [2, 3], [3, 4], [4, 1]])
        
        # Test all algorithms return dict with no exceptions
        algorithms = [
            lambda: shortest_path("test_consistency", 1, 3),
            lambda: connected_components("test_consistency"),
            lambda: centrality_measures("test_consistency"),
            lambda: clustering_coefficients("test_consistency"),
            lambda: minimum_spanning_tree("test_consistency"),
            lambda: graph_coloring("test_consistency"),
            lambda: cycles_detection("test_consistency"),
            lambda: matching("test_consistency"),
            lambda: graph_statistics("test_consistency"),
        ]
        
        for i, algo in enumerate(algorithms):
            result = algo()
            assert isinstance(result, dict), f"Algorithm {i} didn't return dict"
            assert "error" not in result, f"Algorithm {i} returned error: {result}"
        
        print("✅ All algorithms return consistent dict format")
    
    def test_performance_on_larger_graphs(self):
        """Test performance with larger graphs."""
        import time
        
        # Create larger graph
        create_graph("test_large", "undirected")
        nodes = list(range(100))
        add_nodes("test_large", nodes)
        
        # Add edges to create connected graph
        edges = []
        for i in range(99):
            edges.append([i, i+1])
        # Add some random edges
        for i in range(0, 100, 10):
            for j in range(i+2, min(i+10, 100)):
                edges.append([i, j])
        add_edges("test_large", edges)
        
        # Test algorithms that might have performance issues
        start = time.time()
        result = centrality_measures("test_large", ["degree"])
        degree_time = time.time() - start
        assert "degree_centrality" in result
        print(f"  Degree centrality on 100 nodes: {degree_time:.3f}s")
        
        start = time.time()
        result = clustering_coefficients("test_large")
        cluster_time = time.time() - start
        assert "average_clustering" in result
        print(f"  Clustering coefficient on 100 nodes: {cluster_time:.3f}s")
        
        # These should complete quickly
        assert degree_time < 1.0, "Degree centrality too slow"
        assert cluster_time < 1.0, "Clustering coefficient too slow"
        
        print("✅ Performance is acceptable for moderate graphs")


def run_integration_tests():
    """Run all integration tests and summarize results."""
    print("=== Running Algorithm Integration Tests ===\n")
    
    test_suite = TestAlgorithmIntegration()
    test_suite.setup_and_teardown()
    
    tests = [
        ("Shortest Path", test_suite.test_shortest_path_algorithms),
        ("Connected Components", test_suite.test_connected_components),
        ("Centrality Measures", test_suite.test_centrality_measures),
        ("Clustering Coefficients", test_suite.test_clustering_coefficients),
        ("Minimum Spanning Tree", test_suite.test_minimum_spanning_tree),
        ("Maximum Flow", test_suite.test_maximum_flow),
        ("Graph Coloring", test_suite.test_graph_coloring),
        ("Community Detection", test_suite.test_community_detection),
        ("Cycle Detection", test_suite.test_cycles_detection),
        ("Matching", test_suite.test_matching),
        ("Graph Statistics", test_suite.test_graph_statistics),
        ("All Pairs Shortest Path", test_suite.test_all_pairs_shortest_path),
        ("Result Consistency", test_suite.test_algorithm_result_consistency),
        ("Performance", test_suite.test_performance_on_larger_graphs),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nTesting {name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All algorithm integration tests passed!")
    else:
        print(f"\n⚠️  {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/brightliu/Coding_Projects/networkx-mcp-server')
    
    success = run_integration_tests()
    sys.exit(0 if success else 1)