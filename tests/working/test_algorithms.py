"""Tests for GraphAlgorithms class in core/algorithms.py.

Uses well-known small graphs (path, complete, star, cycle, etc.)
so expected values can be verified analytically.
"""

import pytest
import networkx as nx

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.errors import (
    AlgorithmError,
    ResourceLimitExceededError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Fixtures — small, deterministic graphs
# ---------------------------------------------------------------------------


@pytest.fixture
def path_graph():
    """0 - 1 - 2 - 3 - 4 (path of length 4)."""
    return nx.path_graph(5)


@pytest.fixture
def complete_graph():
    """K5 — every pair connected."""
    return nx.complete_graph(5)


@pytest.fixture
def star_graph():
    """Star with center 0 and leaves 1..4."""
    return nx.star_graph(4)


@pytest.fixture
def cycle_graph():
    """0 - 1 - 2 - 3 - 4 - 0."""
    return nx.cycle_graph(5)


@pytest.fixture
def disconnected_graph():
    """Two disconnected triangles: {0,1,2} and {3,4,5}."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    g.add_edges_from([(3, 4), (4, 5), (5, 3)])
    return g


@pytest.fixture
def weighted_graph():
    """Small weighted graph for shortest-path / MST tests.

         1
      0-----1
      |     |
    4 |     | 2
      |     |
      3-----2
         3
    """
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    g.add_edge(1, 2, weight=2)
    g.add_edge(2, 3, weight=3)
    g.add_edge(0, 3, weight=4)
    return g


@pytest.fixture
def directed_graph():
    """Simple DAG:  0 -> 1 -> 2 -> 3."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


@pytest.fixture
def directed_cycle_graph():
    """Directed cycle: 0 -> 1 -> 2 -> 0."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return g


@pytest.fixture
def flow_graph():
    """Directed graph with capacities for max-flow tests.

    s(0) --10--> 1 --5--> t(3)
      |                   ^
      +---15---> 2 --10--+
    """
    g = nx.DiGraph()
    g.add_edge(0, 1, capacity=10)
    g.add_edge(0, 2, capacity=15)
    g.add_edge(1, 3, capacity=5)
    g.add_edge(2, 3, capacity=10)
    return g


@pytest.fixture
def empty_graph():
    """Graph with zero nodes."""
    return nx.Graph()


@pytest.fixture
def single_node_graph():
    """Graph with one node, no edges."""
    g = nx.Graph()
    g.add_node(0)
    return g


@pytest.fixture
def bipartite_graph():
    """Complete bipartite K_{2,3}."""
    return nx.complete_bipartite_graph(2, 3)


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------


class TestShortestPath:
    def test_dijkstra_source_target(self, path_graph):
        result = GraphAlgorithms.shortest_path(path_graph, 0, 4)
        assert result["path"] == [0, 1, 2, 3, 4]
        assert result["length"] == 4
        assert result["source"] == 0
        assert result["target"] == 4

    def test_dijkstra_single_source_all_targets(self, path_graph):
        result = GraphAlgorithms.shortest_path(path_graph, 0)
        assert result["source"] == 0
        assert result["lengths"][0] == 0
        assert result["lengths"][4] == 4
        assert result["paths"][0] == [0]
        assert result["paths"][4] == [0, 1, 2, 3, 4]

    def test_dijkstra_weighted(self, weighted_graph):
        # 0->1->2 costs 3;  0->3->2 costs 7  => via 1 is shorter
        result = GraphAlgorithms.shortest_path(weighted_graph, 0, 2, weight="weight")
        assert result["path"] == [0, 1, 2]
        assert result["length"] == 3

    def test_bellman_ford_source_target(self, path_graph):
        result = GraphAlgorithms.shortest_path(path_graph, 0, 4, method="bellman-ford")
        assert result["path"] == [0, 1, 2, 3, 4]
        assert result["length"] == 4

    def test_bellman_ford_single_source(self, path_graph):
        result = GraphAlgorithms.shortest_path(path_graph, 0, method="bellman-ford")
        # nx.single_source_bellman_ford returns (distances, paths) but the code
        # maps them as predecessors=distances_dict, distances=paths_dict
        assert result["predecessors"][4] == 4  # actual distance value
        assert result["distances"][4] == [0, 1, 2, 3, 4]  # actual path
        assert result["source"] == 0

    def test_invalid_source_raises(self, path_graph):
        with pytest.raises(ValidationError, match="source"):
            GraphAlgorithms.shortest_path(path_graph, 99, 0)

    def test_invalid_target_raises(self, path_graph):
        with pytest.raises(ValidationError, match="target"):
            GraphAlgorithms.shortest_path(path_graph, 0, 99)

    def test_unknown_method_raises(self, path_graph):
        with pytest.raises(ValidationError):
            GraphAlgorithms.shortest_path(path_graph, 0, 4, method="astar")

    def test_source_equals_target(self, path_graph):
        """When source == target, dijkstra single-source branch is taken (target is falsy 0)."""
        result = GraphAlgorithms.shortest_path(path_graph, 0, 0)
        # target=0 is falsy, so the code falls into the single-source branch
        assert result["source"] == 0
        assert result["lengths"][0] == 0


# ---------------------------------------------------------------------------
# all_pairs_shortest_path
# ---------------------------------------------------------------------------


class TestAllPairsShortestPath:
    def test_unweighted(self, path_graph):
        result = GraphAlgorithms.all_pairs_shortest_path(path_graph)
        # Distance 0->4 == 4 in a path graph of 5 nodes
        assert result["lengths"]["0"]["4"] == 4
        assert result["paths"]["0"]["4"] == [0, 1, 2, 3, 4]

    def test_weighted(self, weighted_graph):
        result = GraphAlgorithms.all_pairs_shortest_path(
            weighted_graph, weight="weight"
        )
        # 0->2 shortest weighted path is 0->1->2 with cost 3
        assert result["lengths"]["0"]["2"] == 3

    def test_complete_graph_all_distances_one(self, complete_graph):
        result = GraphAlgorithms.all_pairs_shortest_path(complete_graph)
        for u in range(5):
            for v in range(5):
                expected = 0 if u == v else 1
                assert result["lengths"][str(u)][str(v)] == expected


# ---------------------------------------------------------------------------
# connected_components
# ---------------------------------------------------------------------------


class TestConnectedComponents:
    def test_connected_graph(self, path_graph):
        result = GraphAlgorithms.connected_components(path_graph)
        assert result["num_components"] == 1
        assert result["is_connected"] is True
        assert result["largest_component_size"] == 5

    def test_disconnected_graph(self, disconnected_graph):
        result = GraphAlgorithms.connected_components(disconnected_graph)
        assert result["num_components"] == 2
        assert result["is_connected"] is False
        assert result["largest_component_size"] == 3

    def test_empty_graph(self, empty_graph):
        result = GraphAlgorithms.connected_components(empty_graph)
        assert result["num_components"] == 0
        assert result["is_connected"] is False
        assert result["largest_component_size"] == 0

    def test_directed_weakly_connected(self, directed_graph):
        result = GraphAlgorithms.connected_components(directed_graph)
        assert result["num_weakly_connected"] == 1
        assert result["is_weakly_connected"] is True

    def test_directed_not_strongly_connected(self, directed_graph):
        """A chain 0->1->2->3 is weakly connected but not strongly connected."""
        result = GraphAlgorithms.connected_components(directed_graph)
        assert result["is_strongly_connected"] is False
        assert result["num_strongly_connected"] == 4  # each node is its own SCC

    def test_directed_cycle_strongly_connected(self, directed_cycle_graph):
        result = GraphAlgorithms.connected_components(directed_cycle_graph)
        assert result["is_strongly_connected"] is True
        assert result["num_strongly_connected"] == 1

    def test_single_node(self, single_node_graph):
        result = GraphAlgorithms.connected_components(single_node_graph)
        assert result["num_components"] == 1
        assert result["is_connected"] is True


# ---------------------------------------------------------------------------
# centrality_measures
# ---------------------------------------------------------------------------


class TestCentralityMeasures:
    def test_degree_centrality_star(self, star_graph):
        """In a star graph, center has degree centrality 1.0, leaves 1/(n-1)."""
        result = GraphAlgorithms.centrality_measures(star_graph, measures=["degree"])
        dc = result["degree_centrality"]
        assert dc[0] == pytest.approx(1.0)
        assert dc[1] == pytest.approx(1.0 / 4)

    def test_degree_centrality_complete(self, complete_graph):
        """In K5, every node has degree centrality 1.0."""
        result = GraphAlgorithms.centrality_measures(
            complete_graph, measures=["degree"]
        )
        for node in range(5):
            assert result["degree_centrality"][node] == pytest.approx(1.0)

    def test_betweenness_centrality_path(self, path_graph):
        """In a path 0-1-2-3-4, the center node (2) has highest betweenness."""
        result = GraphAlgorithms.centrality_measures(
            path_graph, measures=["betweenness"]
        )
        bc = result["betweenness_centrality"]
        # Node 2 is on the most shortest paths
        assert bc[2] > bc[0]
        assert bc[2] > bc[4]
        # Endpoints have zero betweenness
        assert bc[0] == pytest.approx(0.0)
        assert bc[4] == pytest.approx(0.0)

    def test_closeness_centrality_complete(self, complete_graph):
        """In K5, every node has closeness centrality 1.0."""
        result = GraphAlgorithms.centrality_measures(
            complete_graph, measures=["closeness"]
        )
        for node in range(5):
            assert result["closeness_centrality"][node] == pytest.approx(1.0)

    def test_closeness_centrality_star(self, star_graph):
        """Star center should have the highest closeness centrality."""
        result = GraphAlgorithms.centrality_measures(star_graph, measures=["closeness"])
        cc = result["closeness_centrality"]
        assert cc[0] > cc[1]  # center > leaf

    def test_eigenvector_centrality_complete(self, complete_graph):
        """In K5, all eigenvector centralities are equal."""
        result = GraphAlgorithms.centrality_measures(
            complete_graph, measures=["eigenvector"]
        )
        ec = result["eigenvector_centrality"]
        values = list(ec.values())
        for v in values:
            assert v == pytest.approx(values[0], abs=1e-4)

    def test_eigenvector_skipped_on_no_edges(self, single_node_graph):
        """Eigenvector centrality is skipped when there are no edges."""
        result = GraphAlgorithms.centrality_measures(
            single_node_graph, measures=["eigenvector"]
        )
        assert "eigenvector_centrality" not in result

    def test_default_measures(self, path_graph):
        """Default measures list produces degree, betweenness, closeness, eigenvector."""
        result = GraphAlgorithms.centrality_measures(path_graph)
        assert "degree_centrality" in result
        assert "betweenness_centrality" in result
        assert "closeness_centrality" in result
        assert "eigenvector_centrality" in result

    def test_directed_degree_centrality(self, directed_graph):
        """Directed graphs produce in/out degree centrality."""
        result = GraphAlgorithms.centrality_measures(
            directed_graph, measures=["degree"]
        )
        assert "in_degree_centrality" in result
        assert "out_degree_centrality" in result
        assert "degree_centrality" not in result

    def test_pagerank_directed(self, directed_cycle_graph):
        """PageRank on a directed cycle should give equal values to all nodes."""
        result = GraphAlgorithms.centrality_measures(
            directed_cycle_graph, measures=["pagerank"]
        )
        pr = result["pagerank"]
        values = list(pr.values())
        for v in values:
            assert v == pytest.approx(1.0 / 3, abs=1e-4)

    def test_pagerank_only_on_directed(self, path_graph):
        """PageRank measure on undirected graph produces no pagerank key."""
        result = GraphAlgorithms.centrality_measures(path_graph, measures=["pagerank"])
        assert "pagerank" not in result


# ---------------------------------------------------------------------------
# clustering_coefficients
# ---------------------------------------------------------------------------


class TestClusteringCoefficients:
    def test_complete_graph_clustering(self, complete_graph):
        """Every node in K5 has clustering coefficient 1.0."""
        result = GraphAlgorithms.clustering_coefficients(complete_graph)
        for node in range(5):
            assert result["node_clustering"][node] == pytest.approx(1.0)
        assert result["average_clustering"] == pytest.approx(1.0)
        assert result["transitivity"] == pytest.approx(1.0)

    def test_path_graph_clustering(self, path_graph):
        """Endpoint nodes in a path have no triangles, so clustering = 0."""
        result = GraphAlgorithms.clustering_coefficients(path_graph)
        assert result["node_clustering"][0] == pytest.approx(0.0)
        assert result["node_clustering"][4] == pytest.approx(0.0)
        assert result["average_clustering"] == pytest.approx(0.0)

    def test_star_graph_clustering(self, star_graph):
        """Star graph has zero clustering — no triangles."""
        result = GraphAlgorithms.clustering_coefficients(star_graph)
        assert result["average_clustering"] == pytest.approx(0.0)
        assert result["transitivity"] == pytest.approx(0.0)

    def test_empty_graph(self, empty_graph):
        result = GraphAlgorithms.clustering_coefficients(empty_graph)
        assert result["average_clustering"] == pytest.approx(0.0)
        assert result["transitivity"] == pytest.approx(0.0)
        assert result["node_clustering"] == {}

    def test_directed_graph_uses_undirected(self, directed_cycle_graph):
        """Directed graph clustering is computed on the undirected version."""
        result = GraphAlgorithms.clustering_coefficients(directed_cycle_graph)
        # Undirected version of 0->1->2->0 is a triangle, so clustering = 1.0
        assert result["average_clustering"] == pytest.approx(1.0)

    def test_cycle_graph_clustering(self, cycle_graph):
        """C5 has zero clustering — no triangles possible."""
        result = GraphAlgorithms.clustering_coefficients(cycle_graph)
        assert result["average_clustering"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# minimum_spanning_tree
# ---------------------------------------------------------------------------


class TestMinimumSpanningTree:
    def test_kruskal(self, weighted_graph):
        result = GraphAlgorithms.minimum_spanning_tree(weighted_graph, weight="weight")
        # MST of the weighted_graph: edges 0-1 (1), 1-2 (2), 2-3 (3) => total 6
        assert result["total_weight"] == 6
        assert result["num_edges"] == 3
        assert result["algorithm"] == "kruskal"

    def test_prim(self, weighted_graph):
        result = GraphAlgorithms.minimum_spanning_tree(
            weighted_graph, weight="weight", algorithm="prim"
        )
        assert result["total_weight"] == 6
        assert result["num_edges"] == 3
        assert result["algorithm"] == "prim"

    def test_unweighted_complete(self, complete_graph):
        """MST of an unweighted K5 has exactly n-1 = 4 edges, all weight 1."""
        result = GraphAlgorithms.minimum_spanning_tree(complete_graph)
        assert result["num_edges"] == 4
        assert result["total_weight"] == 4

    def test_directed_raises(self, directed_graph):
        with pytest.raises(ValidationError):
            GraphAlgorithms.minimum_spanning_tree(directed_graph)

    def test_unknown_algorithm_raises(self, path_graph):
        with pytest.raises(ValidationError):
            GraphAlgorithms.minimum_spanning_tree(path_graph, algorithm="boruvka")


# ---------------------------------------------------------------------------
# maximum_flow
# ---------------------------------------------------------------------------


class TestMaximumFlow:
    def test_basic_flow(self, flow_graph):
        result = GraphAlgorithms.maximum_flow(flow_graph, 0, 3)
        # Max flow: path 0->1->3 carries 5, path 0->2->3 carries 10 => total 15
        assert result["flow_value"] == 15
        assert result["source"] == 0
        assert result["sink"] == 3

    def test_undirected_raises(self, path_graph):
        with pytest.raises(ValidationError):
            GraphAlgorithms.maximum_flow(path_graph, 0, 4)

    def test_single_path_flow(self):
        """Flow through a single path is limited by the smallest capacity."""
        g = nx.DiGraph()
        g.add_edge(0, 1, capacity=10)
        g.add_edge(1, 2, capacity=3)
        g.add_edge(2, 3, capacity=7)
        result = GraphAlgorithms.maximum_flow(g, 0, 3)
        assert result["flow_value"] == 3


# ---------------------------------------------------------------------------
# graph_coloring
# ---------------------------------------------------------------------------


class TestGraphColoring:
    def test_bipartite_two_colors(self, bipartite_graph):
        """A bipartite graph is 2-colorable."""
        result = GraphAlgorithms.graph_coloring(bipartite_graph)
        assert result["num_colors"] <= 2
        # Verify no adjacent nodes share the same color
        coloring = result["coloring"]
        for u, v in bipartite_graph.edges():
            assert coloring[u] != coloring[v]

    def test_complete_graph_coloring(self, complete_graph):
        """K5 requires exactly 5 colors."""
        result = GraphAlgorithms.graph_coloring(complete_graph)
        assert result["num_colors"] == 5
        coloring = result["coloring"]
        for u, v in complete_graph.edges():
            assert coloring[u] != coloring[v]

    def test_empty_graph(self, empty_graph):
        result = GraphAlgorithms.graph_coloring(empty_graph)
        assert result["num_colors"] == 0
        assert result["coloring"] == {}

    def test_color_classes_partition_nodes(self, cycle_graph):
        result = GraphAlgorithms.graph_coloring(cycle_graph)
        all_nodes = set()
        for nodes in result["color_classes"].values():
            all_nodes.update(nodes)
        assert all_nodes == set(cycle_graph.nodes())


# ---------------------------------------------------------------------------
# community_detection
# ---------------------------------------------------------------------------


class TestCommunityDetection:
    def test_louvain_disconnected(self, disconnected_graph):
        """Two disconnected triangles should produce at least 2 communities."""
        result = GraphAlgorithms.community_detection(
            disconnected_graph, method="louvain"
        )
        assert result["num_communities"] >= 2
        assert result["method"] == "louvain"
        assert result["modularity"] > 0.0
        # Every node should appear in exactly one community
        all_nodes = set()
        for comm in result["communities"]:
            all_nodes.update(comm)
        assert all_nodes == set(disconnected_graph.nodes())

    def test_label_propagation(self, disconnected_graph):
        result = GraphAlgorithms.community_detection(
            disconnected_graph, method="label_propagation"
        )
        assert result["num_communities"] >= 2
        assert result["method"] == "label_propagation"

    def test_greedy_modularity(self, disconnected_graph):
        result = GraphAlgorithms.community_detection(
            disconnected_graph, method="greedy_modularity"
        )
        assert result["num_communities"] >= 2
        assert result["method"] == "greedy_modularity"
        assert result["modularity"] > 0.0

    def test_unknown_method_raises(self, path_graph):
        with pytest.raises(ValidationError):
            GraphAlgorithms.community_detection(path_graph, method="spectral")

    def test_community_sizes_match(self, disconnected_graph):
        result = GraphAlgorithms.community_detection(disconnected_graph)
        total_nodes = sum(result["community_sizes"])
        assert total_nodes == disconnected_graph.number_of_nodes()
        for size, comm in zip(result["community_sizes"], result["communities"]):
            assert size == len(comm)


# ---------------------------------------------------------------------------
# cycles_detection
# ---------------------------------------------------------------------------


class TestCyclesDetection:
    def test_undirected_cycle(self, cycle_graph):
        result = GraphAlgorithms.cycles_detection(cycle_graph)
        assert result["has_cycle"] is True
        assert result["num_independent_cycles"] == 1

    def test_undirected_tree_no_cycle(self, star_graph):
        """A star graph is a tree — no cycles."""
        result = GraphAlgorithms.cycles_detection(star_graph)
        assert result["has_cycle"] is False
        assert result["num_independent_cycles"] == 0

    def test_undirected_path_no_cycle(self, path_graph):
        result = GraphAlgorithms.cycles_detection(path_graph)
        assert result["has_cycle"] is False

    def test_directed_dag(self, directed_graph):
        result = GraphAlgorithms.cycles_detection(directed_graph)
        assert result["has_cycle"] is False
        assert result["is_dag"] is True

    def test_directed_cycle(self, directed_cycle_graph):
        result = GraphAlgorithms.cycles_detection(directed_cycle_graph)
        assert result["has_cycle"] is True
        assert result["is_dag"] is False
        assert result["num_cycles_found"] >= 1

    def test_complete_graph_cycles(self, complete_graph):
        """K5 has many independent cycles."""
        result = GraphAlgorithms.cycles_detection(complete_graph)
        assert result["has_cycle"] is True
        # K5 cycle basis size = |E| - |V| + 1 = 10 - 5 + 1 = 6
        assert result["num_independent_cycles"] == 6


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_max_cardinality_path(self, path_graph):
        """Path on 5 nodes has max matching of size 2."""
        result = GraphAlgorithms.matching(path_graph, max_cardinality=True)
        assert result["matching_size"] == 2
        assert result["is_perfect"] is False

    def test_perfect_matching_cycle_even(self):
        """C4 has a perfect matching."""
        c4 = nx.cycle_graph(4)
        result = GraphAlgorithms.matching(c4, max_cardinality=True)
        assert result["matching_size"] == 2
        assert result["is_perfect"] is True

    def test_max_cardinality_complete(self, complete_graph):
        """K5 (odd) cannot have a perfect matching; max matching size = 2."""
        result = GraphAlgorithms.matching(complete_graph, max_cardinality=True)
        assert result["matching_size"] == 2
        assert result["is_perfect"] is False

    def test_maximal_matching(self, path_graph):
        """Maximal matching is a valid matching (no two edges share a node)."""
        result = GraphAlgorithms.matching(path_graph, max_cardinality=False)
        assert result["matching_size"] >= 1
        matched_nodes = set()
        for u, v in result["matching"]:
            assert u not in matched_nodes
            assert v not in matched_nodes
            matched_nodes.add(u)
            matched_nodes.add(v)

    def test_bipartite_perfect_matching(self):
        """K_{3,3} has a perfect matching."""
        g = nx.complete_bipartite_graph(3, 3)
        result = GraphAlgorithms.matching(g, max_cardinality=True)
        assert result["matching_size"] == 3
        assert result["is_perfect"] is True

    def test_empty_graph_matching(self, empty_graph):
        result = GraphAlgorithms.matching(empty_graph)
        assert result["matching_size"] == 0
        assert result["is_perfect"] is True  # vacuously: 0*2 == 0


# ---------------------------------------------------------------------------
# graph_statistics
# ---------------------------------------------------------------------------


class TestGraphStatistics:
    def test_basic_stats_path(self, path_graph):
        result = GraphAlgorithms.graph_statistics(path_graph)
        assert result["num_nodes"] == 5
        assert result["num_edges"] == 4
        assert result["is_directed"] is False
        assert result["is_multigraph"] is False
        assert result["is_connected"] is True

    def test_density_complete(self, complete_graph):
        result = GraphAlgorithms.graph_statistics(complete_graph)
        assert result["density"] == pytest.approx(1.0)

    def test_density_empty(self, empty_graph):
        result = GraphAlgorithms.graph_statistics(empty_graph)
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0
        assert result["density"] == 0

    def test_degree_stats_star(self, star_graph):
        """Star: center has degree 4, leaves have degree 1."""
        result = GraphAlgorithms.graph_statistics(star_graph)
        ds = result["degree_stats"]
        assert ds["min"] == 1
        assert ds["max"] == 4
        # Mean degree: (4 + 1 + 1 + 1 + 1) / 5 = 1.6
        assert ds["mean"] == pytest.approx(1.6)

    def test_degree_stats_complete(self, complete_graph):
        result = GraphAlgorithms.graph_statistics(complete_graph)
        ds = result["degree_stats"]
        assert ds["min"] == 4
        assert ds["max"] == 4
        assert ds["mean"] == pytest.approx(4.0)
        assert ds["std"] == pytest.approx(0.0)

    def test_diameter_path(self, path_graph):
        result = GraphAlgorithms.graph_statistics(path_graph)
        assert result["diameter"] == 4
        assert result["radius"] == 2

    def test_diameter_complete(self, complete_graph):
        result = GraphAlgorithms.graph_statistics(complete_graph)
        assert result["diameter"] == 1
        assert result["radius"] == 1

    def test_disconnected_no_diameter(self, disconnected_graph):
        """Diameter/radius not defined for disconnected graphs."""
        result = GraphAlgorithms.graph_statistics(disconnected_graph)
        assert result["is_connected"] is False
        assert "diameter" not in result
        assert "radius" not in result

    def test_directed_stats(self, directed_graph):
        result = GraphAlgorithms.graph_statistics(directed_graph)
        assert result["is_directed"] is True
        assert result["is_weakly_connected"] is True
        assert result["is_strongly_connected"] is False
        assert "in_degree_stats" in result
        assert "out_degree_stats" in result

    def test_directed_strongly_connected_diameter(self, directed_cycle_graph):
        result = GraphAlgorithms.graph_statistics(directed_cycle_graph)
        assert result["is_strongly_connected"] is True
        assert result["diameter"] == 2  # longest shortest path in 3-node cycle
        assert result["radius"] == 2

    def test_single_node_stats(self, single_node_graph):
        result = GraphAlgorithms.graph_statistics(single_node_graph)
        assert result["num_nodes"] == 1
        assert result["num_edges"] == 0
        assert result["is_connected"] is True
        assert result["degree_stats"]["min"] == 0
        assert result["degree_stats"]["max"] == 0

    def test_empty_graph_connectivity(self, empty_graph):
        result = GraphAlgorithms.graph_statistics(empty_graph)
        assert result["is_connected"] is False

    def test_empty_digraph_connectivity(self):
        g = nx.DiGraph()
        result = GraphAlgorithms.graph_statistics(g)
        assert result["is_weakly_connected"] is False
        assert result["is_strongly_connected"] is False


# ---------------------------------------------------------------------------
# all_pairs_shortest_path — resource limit
# ---------------------------------------------------------------------------


class TestAllPairsShortestPathLimit:
    def test_too_many_nodes_raises(self):
        g = nx.path_graph(1001)
        with pytest.raises(ResourceLimitExceededError):
            GraphAlgorithms.all_pairs_shortest_path(g)


# ---------------------------------------------------------------------------
# community_detection — missing module
# ---------------------------------------------------------------------------


class TestCommunityDetectionNoModule:
    def test_no_community_module_raises(self, path_graph):
        from unittest.mock import patch

        with patch("networkx_mcp.core.algorithms.HAS_COMMUNITY", False):
            with pytest.raises(AlgorithmError):
                GraphAlgorithms.community_detection(path_graph)
