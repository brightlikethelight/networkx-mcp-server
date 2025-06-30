"""Tests for Phase 2 Advanced Analytics features."""

import networkx as nx
import pytest

from networkx_mcp.advanced import (
    BipartiteAnalysis,
    CommunityDetection,
    DirectedAnalysis,
    GraphGenerators,
    MLIntegration,
    NetworkFlow,
    RobustnessAnalysis,
    SpecializedAlgorithms,
)


class TestCommunityDetection:
    """Test community detection algorithms."""

    def test_auto_algorithm_selection(self):
        """Test automatic algorithm selection based on graph size."""
        # Small graph
        small_graph = nx.karate_club_graph()
        result = CommunityDetection.detect_communities(small_graph, algorithm="auto")
        assert result["algorithm_used"] in ["girvan_newman", "spectral"]
        assert result["num_communities"] > 0

        # Medium graph
        medium_graph = nx.barabasi_albert_graph(500, 3)
        result = CommunityDetection.detect_communities(medium_graph, algorithm="auto")
        assert result["algorithm_used"] == "louvain"

        # Large graph
        large_graph = nx.barabasi_albert_graph(15000, 2)
        result = CommunityDetection.detect_communities(large_graph, algorithm="auto")
        assert result["algorithm_used"] == "label_propagation"

    def test_community_quality_metrics(self):
        """Test community quality assessment."""
        graph = nx.karate_club_graph()
        # Use actual community detection to get a valid partition
        detected = CommunityDetection.detect_communities(graph, algorithm="louvain")
        communities = detected["communities"]

        result = CommunityDetection.community_quality(graph, communities)
        assert "modularity" in result
        assert "coverage" in result
        assert "performance" in result
        assert 0 <= result["modularity"] <= 1

    def test_hierarchical_communities(self):
        """Test hierarchical community detection."""
        graph = nx.karate_club_graph()
        result = CommunityDetection.hierarchical_communities(graph, max_levels=3)

        assert len(result["levels"]) <= 3
        # With increasing resolution, we expect more communities
        assert (
            result["levels"][0]["num_communities"]
            <= result["levels"][-1]["num_communities"]
        )


class TestNetworkFlow:
    """Test network flow algorithms."""

    def test_max_flow_auto_selection(self):
        """Test automatic flow algorithm selection."""
        # Small graph
        G = nx.DiGraph()
        G.add_edges_from([(0, 1, {"capacity": 10}), (1, 2, {"capacity": 5})])
        result = NetworkFlow.max_flow_analysis(G, 0, 2, algorithm="auto")
        assert result["algorithm_used"] == "edmonds_karp"
        assert result["max_flow_value"] == 5

    def test_min_cut_analysis(self):
        """Test minimum cut analysis."""
        G = nx.DiGraph()
        edges = [
            (0, 1, 10),
            (0, 2, 10),
            (1, 2, 2),
            (1, 3, 4),
            (1, 4, 8),
            (2, 4, 9),
            (3, 5, 10),
            (4, 3, 6),
            (4, 5, 10),
        ]
        for u, v, c in edges:
            G.add_edge(u, v, capacity=c)

        result = NetworkFlow.min_cut_analysis(G, 0, 5)
        assert "min_cut_value" in result
        assert "cut_edges" in result
        assert result["min_cut_value"] > 0


class TestGraphGenerators:
    """Test graph generation algorithms."""

    def test_scale_free_generation(self):
        """Test scale-free graph generation."""
        result = GraphGenerators.scale_free_graph(100, m=3, model="barabasi_albert")

        assert result["graph"].number_of_nodes() == 100
        assert result["graph"].number_of_edges() > 0
        assert "power_law_exponent" in result["properties"]
        assert 2.0 <= result["properties"]["power_law_exponent"] <= 3.5

    def test_small_world_generation(self):
        """Test small-world graph generation."""
        result = GraphGenerators.small_world_graph(100, k=6, p=0.3)

        graph = result["graph"]
        assert graph.number_of_nodes() == 100
        assert "clustering_coefficient" in result["properties"]
        assert "average_shortest_path" in result["properties"]
        assert result["properties"]["small_world_sigma"] > 1

    def test_social_network_generation(self):
        """Test social network model generation."""
        result = GraphGenerators.social_network_graph(
            100, communities=5, p_in=0.3, p_out=0.05, model="stochastic_block"
        )

        assert result["graph"].number_of_nodes() == 100
        assert result["properties"]["num_communities"] == 5
        assert result["properties"]["modularity"] > 0.3


class TestBipartiteAnalysis:
    """Test bipartite graph analysis."""

    def test_bipartite_check(self):
        """Test bipartite graph detection."""
        # Create bipartite graph
        B = nx.Graph()
        B.add_edges_from([(0, "a"), (0, "b"), (1, "a"), (1, "c"), (2, "b")])

        result = BipartiteAnalysis.is_bipartite(B)
        assert result["is_bipartite"]
        assert len(result["node_sets"]) == 2

        # Non-bipartite graph
        G = nx.cycle_graph(5)
        result = BipartiteAnalysis.is_bipartite(G)
        assert not result["is_bipartite"]

    def test_maximum_matching(self):
        """Test maximum matching in bipartite graph."""
        B = nx.Graph()
        B.add_edges_from([("A", 1), ("A", 2), ("B", 2), ("B", 3), ("C", 3), ("C", 4)])

        result = BipartiteAnalysis.maximum_matching(B)
        assert result["matching_size"] == 3
        assert len(result["matching"]) == 3


class TestDirectedAnalysis:
    """Test directed graph analysis."""

    def test_dag_analysis(self):
        """Test DAG analysis."""
        # Create DAG
        G = nx.DiGraph([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])

        result = DirectedAnalysis.dag_analysis(G)
        assert result["is_dag"]
        assert result["topological_generations"] is not None
        assert result["longest_path_length"] == 3

        # Add cycle
        G.add_edge(5, 1)
        result = DirectedAnalysis.dag_analysis(G)
        assert not result["is_dag"]

    def test_bow_tie_structure(self):
        """Test bow-tie decomposition."""
        # Create bow-tie structure
        G = nx.DiGraph()
        # SCC
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        # IN component
        G.add_edges_from([(4, 1), (5, 4)])
        # OUT component
        G.add_edges_from([(3, 6), (6, 7)])
        # Tendril from IN
        G.add_edge(5, 8)
        # Disconnected
        G.add_edge(9, 10)

        result = DirectedAnalysis.bow_tie_structure(G)
        assert result["components"]["scc"]["size"] == 3
        assert result["components"]["in"]["size"] == 2
        assert result["components"]["out"]["size"] == 2


class TestSpecializedAlgorithms:
    """Test specialized algorithms."""

    def test_graph_coloring(self):
        """Test graph coloring algorithms."""
        G = nx.petersen_graph()

        result = SpecializedAlgorithms.graph_coloring(G, strategy="dsatur")
        assert result["is_valid_coloring"]
        assert result["num_colors_used"] == 3  # Petersen graph chromatic number

        # Verify coloring
        coloring = result["coloring"]
        for u, v in G.edges():
            assert coloring[u] != coloring[v]

    def test_link_prediction(self):
        """Test link prediction algorithms."""
        G = nx.karate_club_graph()
        # Remove some edges
        edges_to_remove = [(0, 2), (1, 3), (5, 7)]
        G.remove_edges_from(edges_to_remove)

        result = SpecializedAlgorithms.link_prediction(G, method="adamic_adar", top_k=5)

        assert len(result["top_predictions"]) <= 5
        assert all("score" in pred for pred in result["top_predictions"])


class TestMLIntegration:
    """Test machine learning integrations."""

    def test_node_embeddings(self):
        """Test node embedding generation."""
        G = nx.karate_club_graph()

        # Test different methods
        for method in ["spectral", "structural"]:
            result = MLIntegration.node_embeddings(G, method=method, dimensions=32)

            assert len(result["embeddings"]) == G.number_of_nodes()
            assert all(len(emb) == 32 for emb in result["embeddings"].values())
            assert result["embedding_stats"]["sparsity"] >= 0

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Create graph with anomalous nodes
        G = nx.barabasi_albert_graph(100, 3)
        # Add isolated node (anomaly)
        G.add_node(100)
        # Add star subgraph (anomaly)
        star_center = 101
        G.add_edges_from([(star_center, i) for i in range(102, 112)])

        result = MLIntegration.anomaly_detection(
            G, method="statistical", contamination=0.1
        )

        assert len(result["anomalous_nodes"]) > 0
        assert result["num_anomalous_nodes"] <= int(G.number_of_nodes() * 0.1 + 1)
        # Check if isolated node is detected
        assert (
            100 in result["anomalous_nodes"] or star_center in result["anomalous_nodes"]
        )


class TestRobustnessAnalysis:
    """Test robustness analysis."""

    def test_attack_simulation(self):
        """Test attack simulation."""
        G = nx.barabasi_albert_graph(100, 3)

        # Random attack
        result = RobustnessAnalysis.attack_simulation(
            G, attack_type="random", fraction=0.2, measure="connectivity"
        )

        assert result["num_nodes_removed"] == 20
        assert result["robustness_index"] >= 0
        assert len(result["removal_sequence"]) > 0

        # Targeted attack
        result_targeted = RobustnessAnalysis.attack_simulation(
            G, attack_type="targeted_degree", fraction=0.2, measure="largest_component"
        )

        # Targeted attacks should be more effective
        assert result_targeted["robustness_index"] < result["robustness_index"]

    def test_percolation_analysis(self):
        """Test percolation analysis."""
        G = nx.erdos_renyi_graph(100, 0.05)

        result = RobustnessAnalysis.percolation_analysis(
            G,
            percolation_type="site",
            probability_range=(0.0, 1.0),
            num_steps=10,
            num_trials=5,
        )

        assert len(result["results"]) == 10
        assert result["percolation_threshold"] is not None
        assert 0 <= result["percolation_threshold"] <= 1

    def test_network_resilience(self):
        """Test comprehensive resilience metrics."""
        G = nx.karate_club_graph()

        result = RobustnessAnalysis.network_resilience(
            G, resilience_metrics=["connectivity", "redundancy", "clustering"]
        )

        assert "overall_resilience_score" in result
        assert 0 <= result["overall_resilience_score"] <= 1
        assert "connectivity" in result
        assert "redundancy" in result
        assert "clustering" in result


# Performance Tests
class TestPerformance:
    """Performance tests for advanced analytics."""

    @pytest.mark.performance
    def test_large_graph_community_detection(self):
        """Test community detection on large graphs."""
        import time

        G = nx.barabasi_albert_graph(10000, 5)

        start = time.time()
        result = CommunityDetection.detect_communities(G, algorithm="auto")
        elapsed = time.time() - start

        assert elapsed < 10.0  # Should complete within 10 seconds
        assert result["num_communities"] > 0

    @pytest.mark.performance
    def test_embedding_scalability(self):
        """Test embedding generation scalability."""
        import time

        sizes = [100, 500, 1000]
        times = []

        for n in sizes:
            G = nx.barabasi_albert_graph(n, 3)

            start = time.time()
            MLIntegration.node_embeddings(G, method="structural", dimensions=16)
            elapsed = time.time() - start
            times.append(elapsed)

        # Check that time scales reasonably
        assert all(t < 10.0 for t in times)  # All should complete within 10s
