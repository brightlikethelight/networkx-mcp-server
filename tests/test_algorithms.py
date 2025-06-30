"""Comprehensive tests for graph algorithms."""

import time

import networkx as nx
import numpy as np
import pytest
from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.utils.monitoring import PerformanceMonitor


class TestGraphAlgorithms:
    """Test GraphAlgorithms class."""

    def test_shortest_path_dijkstra(self):
        """Test Dijkstra's shortest path algorithm."""
        # Create weighted graph
        G = nx.Graph()
        G.add_weighted_edges_from(
            [("A", "B", 1), ("B", "C", 2), ("A", "C", 4), ("C", "D", 1), ("B", "D", 5)]
        )

        # Single target
        result = GraphAlgorithms.shortest_path(G, "A", "D", weight="weight")
        assert result["path"] == ["A", "B", "C", "D"]
        assert result["length"] == 4

        # All targets from source
        result = GraphAlgorithms.shortest_path(G, "A", weight="weight")
        assert result["paths"]["A"] == ["A"]
        assert result["paths"]["D"] == ["A", "B", "C", "D"]
        assert result["lengths"]["D"] == 4

    def test_shortest_path_unweighted(self):
        """Test shortest path on unweighted graph."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")])

        result = GraphAlgorithms.shortest_path(G, "A", "C")
        assert result["path"] == ["A", "B", "C"]
        assert result["length"] == 2

    def test_all_pairs_shortest_path(self):
        """Test all pairs shortest path."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        result = GraphAlgorithms.all_pairs_shortest_path(G)
        assert "A" in result["lengths"]
        assert result["lengths"]["A"]["C"] == 2
        assert result["paths"]["A"]["C"] == ["A", "B", "C"]

    def test_connected_components_undirected(self):
        """Test connected components for undirected graph."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("D", "E")])

        result = GraphAlgorithms.connected_components(G)
        assert result["num_components"] == 2
        assert result["is_connected"] is False
        components = result["connected_components"]
        assert len(components) == 2
        assert {"A", "B", "C"} in [set(comp) for comp in components]
        assert {"D", "E"} in [set(comp) for comp in components]

    def test_connected_components_directed(self):
        """Test connected components for directed graph."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])

        result = GraphAlgorithms.connected_components(G)
        assert result["is_weakly_connected"] is False
        assert result["is_strongly_connected"] is False
        assert result["num_weakly_connected"] == 2
        assert result["num_strongly_connected"] == 3

    def test_centrality_measures(self):
        """Test various centrality measures."""
        G = nx.karate_club_graph()

        result = GraphAlgorithms.centrality_measures(
            G, measures=["degree", "betweenness", "closeness"]
        )

        assert "degree_centrality" in result
        assert "betweenness_centrality" in result
        assert "closeness_centrality" in result

        # Check that values are computed
        assert len(result["degree_centrality"]) == G.number_of_nodes()
        assert all(0 <= v <= 1 for v in result["degree_centrality"].values())

    def test_clustering_coefficients(self):
        """Test clustering coefficient calculation."""
        G = nx.complete_graph(4)

        result = GraphAlgorithms.clustering_coefficients(G)
        assert "node_clustering" in result
        assert "average_clustering" in result
        assert "transitivity" in result

        # Complete graph has clustering coefficient 1
        assert result["average_clustering"] == 1.0
        assert all(v == 1.0 for v in result["node_clustering"].values())

    def test_minimum_spanning_tree(self):
        """Test minimum spanning tree algorithms."""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [("A", "B", 1), ("B", "C", 2), ("A", "C", 3), ("B", "D", 4), ("C", "D", 1)]
        )

        # Kruskal's algorithm
        result = GraphAlgorithms.minimum_spanning_tree(G, algorithm="kruskal")
        assert result["num_edges"] == 3
        assert result["total_weight"] == 4  # Edges: AB(1), BC(2), CD(1)

        # Prim's algorithm
        result = GraphAlgorithms.minimum_spanning_tree(G, algorithm="prim")
        assert result["num_edges"] == 3
        assert result["total_weight"] == 4

    def test_maximum_flow(self):
        """Test maximum flow algorithm."""
        G = nx.DiGraph()
        G.add_edge("s", "a", capacity=3)
        G.add_edge("s", "b", capacity=2)
        G.add_edge("a", "b", capacity=1)
        G.add_edge("a", "t", capacity=2)
        G.add_edge("b", "t", capacity=2)

        result = GraphAlgorithms.maximum_flow(G, "s", "t")
        assert result["flow_value"] == 4
        assert result["source"] == "s"
        assert result["sink"] == "t"

    def test_graph_coloring(self):
        """Test graph coloring."""
        G = nx.cycle_graph(5)  # 5-cycle needs 3 colors

        result = GraphAlgorithms.graph_coloring(G)
        assert "coloring" in result
        assert "num_colors" in result

        # Verify valid coloring
        coloring = result["coloring"]
        for u, v in G.edges():
            assert coloring[u] != coloring[v]

    def test_community_detection(self):
        """Test community detection algorithms."""
        G = nx.karate_club_graph()

        # Louvain method
        result = GraphAlgorithms.community_detection(G, method="louvain")
        assert "communities" in result
        assert "modularity" in result
        assert result["num_communities"] > 0

        # Check all nodes are in communities
        all_nodes = set()
        for comm in result["communities"]:
            all_nodes.update(comm)
        assert len(all_nodes) == G.number_of_nodes()

    def test_cycles_detection_directed(self):
        """Test cycle detection in directed graphs."""
        # DAG - no cycles
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

        result = GraphAlgorithms.cycles_detection(G)
        assert result["has_cycle"] is False
        assert result["is_dag"] is True

        # Add cycle
        G.add_edge("C", "A")
        result = GraphAlgorithms.cycles_detection(G)
        assert result["has_cycle"] is True
        assert result["is_dag"] is False
        assert len(result["cycles"]) > 0

    def test_cycles_detection_undirected(self):
        """Test cycle detection in undirected graphs."""
        # Tree - no cycles
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("B", "D")])

        result = GraphAlgorithms.cycles_detection(G)
        assert result["has_cycle"] is False
        assert result["num_independent_cycles"] == 0

        # Add cycle
        G.add_edge("C", "D")
        result = GraphAlgorithms.cycles_detection(G)
        assert result["has_cycle"] is True
        assert result["num_independent_cycles"] == 1

    def test_matching(self):
        """Test matching algorithms."""
        G = nx.Graph()
        G.add_edges_from(
            [("A", "1"), ("A", "2"), ("B", "2"), ("B", "3"), ("C", "3"), ("C", "4")]
        )

        result = GraphAlgorithms.matching(G, max_cardinality=True)
        assert "matching" in result
        assert "matching_size" in result

        # Verify valid matching
        matched_nodes = set()
        for u, v in result["matching"]:
            assert u not in matched_nodes
            assert v not in matched_nodes
            matched_nodes.update([u, v])

    def test_graph_statistics(self):
        """Test graph statistics calculation."""
        G = nx.karate_club_graph()

        stats = GraphAlgorithms.graph_statistics(G)
        assert stats["num_nodes"] == 34
        assert stats["num_edges"] == 78
        assert "density" in stats
        assert "degree_stats" in stats
        assert stats["is_connected"] is True
        assert "diameter" in stats
        assert "radius" in stats

        # Test directed graph stats
        D = nx.DiGraph(G)
        stats = GraphAlgorithms.graph_statistics(D)
        assert stats["is_directed"] is True
        assert "in_degree_stats" in stats
        assert "out_degree_stats" in stats

    def test_invalid_inputs(self):
        """Test algorithm behavior with invalid inputs."""
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C"])

        # Shortest path with non-existent node
        with pytest.raises(ValueError):
            GraphAlgorithms.shortest_path(G, "X", "A")

        # Maximum flow on undirected graph
        with pytest.raises(ValueError):
            GraphAlgorithms.maximum_flow(G, "A", "B")

        # MST on directed graph
        D = nx.DiGraph()
        D.add_edge("A", "B")
        with pytest.raises(ValueError):
            GraphAlgorithms.minimum_spanning_tree(D)


class TestAlgorithmPerformance:
    """Test algorithm performance on various graph sizes."""

    def setup_method(self):
        """Set up performance monitoring."""
        self.monitor = PerformanceMonitor()

    def test_centrality_performance(self):
        """Test centrality calculation performance."""
        # Test on graphs of increasing size
        sizes = [10, 50, 100, 200]
        times = []

        for n in sizes:
            # Create random graph
            G = nx.fast_gnp_random_graph(n, 0.1)

            start = time.time()
            centrality = GraphAlgorithms.centrality_measures(
                G, measures=["degree", "betweenness"]
            )
            elapsed = time.time() - start

            times.append(elapsed)
            self.monitor.record_operation(f"centrality_n{n}", elapsed)

            # Verify results
            assert "degree_centrality" in centrality
            assert "betweenness_centrality" in centrality
            assert len(centrality["degree_centrality"]) == n

        # Check performance scaling
        # Time should not grow too rapidly (allow for larger tolerance)
        assert (
            times[-1] < times[0] * 100 or times[-1] < 1.0
        )  # Reasonable scaling or under 1 second

    def test_shortest_path_performance(self):
        """Test shortest path performance."""
        # Create a grid graph (known structure)
        G = nx.grid_2d_graph(10, 10)

        # Convert node labels to strings
        mapping = {node: f"{node[0]},{node[1]}" for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        start = time.time()
        # Find paths from corner to corner
        result = GraphAlgorithms.shortest_path(G, "0,0", "9,9")
        elapsed = time.time() - start

        assert result["path"] is not None
        assert result["length"] == 18  # Manhattan distance
        assert elapsed < 0.1  # Should be very fast

    def test_component_performance(self):
        """Test connected components performance."""
        # Create graph with multiple components
        G = nx.Graph()

        # Add several disconnected cliques
        for i in range(10):
            nodes = [f"C{i}N{j}" for j in range(20)]
            G.add_nodes_from(nodes)
            # Make complete subgraph
            for j in range(len(nodes)):
                for k in range(j + 1, len(nodes)):
                    G.add_edge(nodes[j], nodes[k])

        start = time.time()
        result = GraphAlgorithms.connected_components(G)
        elapsed = time.time() - start

        assert result["num_components"] == 10
        assert result["is_connected"] is False
        assert elapsed < 0.5  # Should handle 200 nodes quickly


class TestAdvancedAlgorithms:
    """Test advanced algorithm features."""

    def test_weighted_shortest_paths(self):
        """Test shortest paths with various weight scenarios."""
        G = nx.DiGraph()

        # Create graph with different weight scenarios
        edges = [
            ("A", "B", {"weight": 1, "cost": 10}),
            ("A", "C", {"weight": 2, "cost": 5}),
            ("B", "D", {"weight": 1, "cost": 3}),
            ("C", "D", {"weight": 1, "cost": 2}),
            ("B", "E", {"weight": 3, "cost": 1}),
            ("D", "E", {"weight": 1, "cost": 1}),
        ]
        G.add_edges_from(edges)

        # Test with different weight attributes
        result_weight = GraphAlgorithms.shortest_path(G, "A", "E", weight="weight")
        result_cost = GraphAlgorithms.shortest_path(G, "A", "E", weight="cost")

        # Different paths based on weight attribute
        assert result_weight["path"] == ["A", "B", "D", "E"]
        assert result_weight["length"] == 3

        assert result_cost["path"] == ["A", "C", "D", "E"]
        assert result_cost["length"] == 8

    def test_negative_weight_detection(self):
        """Test handling of negative weights."""
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [
                ("A", "B", 1),
                ("B", "C", -2),  # Negative weight
                ("C", "D", 1),
                ("B", "D", 3),
            ]
        )

        # Bellman-Ford should handle negative weights
        result = GraphAlgorithms.shortest_path(
            G, "A", "D", weight="weight", method="bellman-ford"
        )

        assert result["path"] == ["A", "B", "C", "D"]
        assert result["length"] == 0  # 1 + (-2) + 1

        # Add negative cycle
        G.add_edge("D", "B", weight=-5)

        # Should detect negative cycle
        with pytest.raises(nx.NetworkXUnbounded):
            GraphAlgorithms.shortest_path(
                G, "A", "D", weight="weight", method="bellman-ford"
            )

    def test_centrality_edge_cases(self):
        """Test centrality measures on special graph structures."""
        # Star graph - one central node
        star = nx.star_graph(10)
        star_centrality = GraphAlgorithms.centrality_measures(
            star, measures=["degree", "betweenness"]
        )

        # Center node (0) should have highest centrality
        degree_cent = star_centrality["degree_centrality"]
        assert degree_cent[0] == 1.0  # Maximum degree centrality
        assert all(degree_cent[i] == 0.1 for i in range(1, 11))

        # Cycle graph - all nodes equal
        cycle = nx.cycle_graph(10)
        cycle_centrality = GraphAlgorithms.centrality_measures(
            cycle, measures=["degree", "closeness"]
        )

        # All nodes should have same centrality
        degree_values = set(cycle_centrality["degree_centrality"].values())
        assert len(degree_values) == 1  # All equal

    def test_community_detection_quality(self):
        """Test community detection on graphs with known structure."""
        # Create graph with clear community structure
        G = nx.Graph()

        # Community 1
        community1 = [f"A{i}" for i in range(10)]
        G.add_nodes_from(community1)
        for i in range(10):
            for j in range(i + 1, 10):
                if np.random.random() < 0.8:  # Dense connections
                    G.add_edge(community1[i], community1[j])

        # Community 2
        community2 = [f"B{i}" for i in range(10)]
        G.add_nodes_from(community2)
        for i in range(10):
            for j in range(i + 1, 10):
                if np.random.random() < 0.8:  # Dense connections
                    G.add_edge(community2[i], community2[j])

        # Few inter-community edges
        for _ in range(3):
            i = np.random.randint(0, 10)
            j = np.random.randint(0, 10)
            G.add_edge(community1[i], community2[j])

        # Detect communities
        result = GraphAlgorithms.community_detection(G, method="louvain")

        assert result["num_communities"] >= 2
        assert result["modularity"] > 0.3  # Good modularity

        # Check if detected communities match expected
        detected_comms = result["communities"]

        # Most nodes from same original community should be together
        for comm in detected_comms:
            a_nodes = sum(1 for node in comm if node.startswith("A"))
            b_nodes = sum(1 for node in comm if node.startswith("B"))
            # One type should dominate
            assert max(a_nodes, b_nodes) > 0.7 * len(comm)


class TestGraphMetrics:
    """Test comprehensive graph metrics calculation."""

    def test_complete_graph_metrics(self):
        """Test metrics on complete graph."""
        K5 = nx.complete_graph(5)
        stats = GraphAlgorithms.graph_statistics(K5)

        assert stats["num_nodes"] == 5
        assert stats["num_edges"] == 10
        assert stats["density"] == 1.0
        assert stats["is_connected"] is True
        assert stats["diameter"] == 1
        assert stats["radius"] == 1

        # Degree stats for complete graph
        assert stats["degree_stats"]["mean"] == 4.0
        assert stats["degree_stats"]["min"] == 4
        assert stats["degree_stats"]["max"] == 4

    def test_path_graph_metrics(self):
        """Test metrics on path graph."""
        P10 = nx.path_graph(10)
        stats = GraphAlgorithms.graph_statistics(P10)

        assert stats["num_nodes"] == 10
        assert stats["num_edges"] == 9
        assert stats["is_connected"] is True
        assert stats["diameter"] == 9
        assert stats["radius"] == 5

        # Degree distribution
        # 2 endpoints with degree 1, others with degree 2
        assert stats["degree_stats"]["min"] == 1
        assert stats["degree_stats"]["max"] == 2

    def test_directed_graph_metrics(self):
        """Test metrics specific to directed graphs."""
        # Create a directed acyclic graph
        D = nx.DiGraph()
        D.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")])

        stats = GraphAlgorithms.graph_statistics(D)

        assert stats["is_directed"] is True
        assert stats["is_strongly_connected"] is False
        assert stats["is_weakly_connected"] is True

        # In/out degree stats
        assert "in_degree_stats" in stats
        assert "out_degree_stats" in stats
