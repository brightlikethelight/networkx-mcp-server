"""Benchmarks for graph algorithms.

This module benchmarks the performance of graph algorithms
implemented in the NetworkX MCP server.
"""

import networkx as nx
import numpy as np


class PathfindingSuite:
    """Benchmark suite for pathfinding algorithms."""

    def setup(self):
        """Set up test graphs for pathfinding benchmarks."""
        # Create graphs of different types and sizes
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(500, 0.05, seed=42)
        self.large_graph = nx.erdos_renyi_graph(1000, 0.02, seed=42)

        # Scale-free graph for realistic networks
        self.scale_free = nx.barabasi_albert_graph(500, 3, seed=42)

        # Grid graph for worst-case scenarios
        self.grid_graph = nx.grid_2d_graph(20, 25)

        # Add weights to edges for weighted algorithms
        for G in [self.small_graph, self.medium_graph, self.large_graph]:
            for u, v in G.edges():
                G.edges[u, v]["weight"] = np.random.uniform(0.1, 10.0)

        # Get node pairs for testing
        self.small_nodes = list(self.small_graph.nodes())
        self.medium_nodes = list(self.medium_graph.nodes())
        self.large_nodes = list(self.large_graph.nodes())

        # Ensure connectivity for reliable benchmarks
        self.small_graph = max(nx.connected_components(self.small_graph), key=len)
        self.small_graph = self.small_graph.subgraph(self.small_graph)

    def time_shortest_path_unweighted_small(self):
        """Benchmark unweighted shortest path on small graph."""
        if len(self.small_nodes) >= 2:
            source, target = self.small_nodes[0], self.small_nodes[-1]
            try:
                return nx.shortest_path(self.small_graph, source, target)
            except nx.NetworkXNoPath:
                return None

    def time_shortest_path_unweighted_medium(self):
        """Benchmark unweighted shortest path on medium graph."""
        if len(self.medium_nodes) >= 2:
            source, target = self.medium_nodes[0], self.medium_nodes[-1]
            try:
                return nx.shortest_path(self.medium_graph, source, target)
            except nx.NetworkXNoPath:
                return None

    def time_shortest_path_weighted_small(self):
        """Benchmark weighted shortest path on small graph."""
        if len(self.small_nodes) >= 2:
            source, target = self.small_nodes[0], self.small_nodes[-1]
            try:
                return nx.shortest_path(
                    self.small_graph, source, target, weight="weight"
                )
            except nx.NetworkXNoPath:
                return None

    def time_shortest_path_weighted_medium(self):
        """Benchmark weighted shortest path on medium graph."""
        if len(self.medium_nodes) >= 2:
            source, target = self.medium_nodes[0], self.medium_nodes[-1]
            try:
                return nx.shortest_path(
                    self.medium_graph, source, target, weight="weight"
                )
            except nx.NetworkXNoPath:
                return None

    def time_all_shortest_paths_small(self):
        """Benchmark all shortest paths on small graph."""
        if len(self.small_nodes) >= 2:
            source, target = self.small_nodes[0], self.small_nodes[-1]
            try:
                return list(nx.all_shortest_paths(self.small_graph, source, target))
            except nx.NetworkXNoPath:
                return []

    def time_dijkstra_single_source_small(self):
        """Benchmark single-source Dijkstra on small graph."""
        if self.small_nodes:
            source = self.small_nodes[0]
            return nx.single_source_dijkstra_path_length(
                self.small_graph, source, weight="weight"
            )

    def time_dijkstra_single_source_medium(self):
        """Benchmark single-source Dijkstra on medium graph."""
        if self.medium_nodes:
            source = self.medium_nodes[0]
            return nx.single_source_dijkstra_path_length(
                self.medium_graph, source, weight="weight"
            )

    def time_bellman_ford_small(self):
        """Benchmark Bellman-Ford algorithm on small graph."""
        if self.small_nodes:
            source = self.small_nodes[0]
            try:
                return nx.bellman_ford_path_length(
                    self.small_graph, source, weight="weight"
                )
            except (nx.NetworkXUnbounded, nx.NodeNotFound):
                return {}

    def peakmem_all_pairs_shortest_path_small(self):
        """Benchmark peak memory for all-pairs shortest path."""
        return dict(nx.all_pairs_shortest_path_length(self.small_graph))


class CentralitySuite:
    """Benchmark suite for centrality algorithms."""

    def setup(self):
        """Set up test graphs for centrality benchmarks."""
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(300, 0.05, seed=42)
        self.scale_free = nx.barabasi_albert_graph(200, 3, seed=42)
        self.small_world = nx.watts_strogatz_graph(200, 6, 0.3, seed=42)

    def time_degree_centrality_small(self):
        """Benchmark degree centrality on small graph."""
        return nx.degree_centrality(self.small_graph)

    def time_degree_centrality_medium(self):
        """Benchmark degree centrality on medium graph."""
        return nx.degree_centrality(self.medium_graph)

    def time_betweenness_centrality_small(self):
        """Benchmark betweenness centrality on small graph."""
        return nx.betweenness_centrality(self.small_graph)

    def time_betweenness_centrality_approximate_medium(self):
        """Benchmark approximate betweenness centrality on medium graph."""
        return nx.betweenness_centrality(self.medium_graph, k=50)

    def time_closeness_centrality_small(self):
        """Benchmark closeness centrality on small graph."""
        return nx.closeness_centrality(self.small_graph)

    def time_eigenvector_centrality_small(self):
        """Benchmark eigenvector centrality on small graph."""
        try:
            return nx.eigenvector_centrality(self.small_graph, max_iter=1000)
        except nx.NetworkXError:
            return {}

    def time_pagerank_small(self):
        """Benchmark PageRank on small graph."""
        return nx.pagerank(self.small_graph)

    def time_pagerank_medium(self):
        """Benchmark PageRank on medium graph."""
        return nx.pagerank(self.medium_graph)

    def time_katz_centrality_small(self):
        """Benchmark Katz centrality on small graph."""
        try:
            return nx.katz_centrality(self.small_graph)
        except nx.NetworkXError:
            return {}

    def peakmem_betweenness_centrality_medium(self):
        """Benchmark peak memory for betweenness centrality."""
        return nx.betweenness_centrality(self.medium_graph)


class CommunityDetectionSuite:
    """Benchmark suite for community detection algorithms."""

    def setup(self):
        """Set up test graphs for community detection."""
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(300, 0.05, seed=42)
        self.scale_free = nx.barabasi_albert_graph(200, 3, seed=42)

        # Create a graph with known community structure
        self.modular_graph = nx.planted_partition_graph(4, 25, 0.8, 0.1, seed=42)

    def time_greedy_modularity_small(self):
        """Benchmark greedy modularity communities on small graph."""
        return list(nx.community.greedy_modularity_communities(self.small_graph))

    def time_greedy_modularity_medium(self):
        """Benchmark greedy modularity communities on medium graph."""
        return list(nx.community.greedy_modularity_communities(self.medium_graph))

    def time_label_propagation_small(self):
        """Benchmark label propagation on small graph."""
        return list(nx.community.label_propagation_communities(self.small_graph))

    def time_label_propagation_medium(self):
        """Benchmark label propagation on medium graph."""
        return list(nx.community.label_propagation_communities(self.medium_graph))

    def time_girvan_newman_small(self):
        """Benchmark Girvan-Newman algorithm on small graph."""
        communities_generator = nx.community.girvan_newman(self.small_graph)
        return next(communities_generator)  # Get first level

    def time_modularity_calculation_small(self):
        """Benchmark modularity calculation on small graph."""
        communities = list(nx.community.greedy_modularity_communities(self.small_graph))
        return nx.community.modularity(self.small_graph, communities)

    def time_modularity_calculation_medium(self):
        """Benchmark modularity calculation on medium graph."""
        communities = list(
            nx.community.greedy_modularity_communities(self.medium_graph)
        )
        return nx.community.modularity(self.medium_graph, communities)

    def peakmem_community_detection_medium(self):
        """Benchmark peak memory for community detection."""
        return list(nx.community.greedy_modularity_communities(self.medium_graph))


class ConnectivitySuite:
    """Benchmark suite for connectivity algorithms."""

    def setup(self):
        """Set up test graphs for connectivity benchmarks."""
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(500, 0.05, seed=42)
        self.directed_graph = nx.erdos_renyi_graph(200, 0.05, seed=42, directed=True)

        # Create disconnected graph
        self.disconnected = nx.union(
            nx.erdos_renyi_graph(50, 0.1, seed=42),
            nx.erdos_renyi_graph(50, 0.1, seed=43),
        )

    def time_connected_components_small(self):
        """Benchmark connected components on small graph."""
        return list(nx.connected_components(self.small_graph))

    def time_connected_components_medium(self):
        """Benchmark connected components on medium graph."""
        return list(nx.connected_components(self.medium_graph))

    def time_strongly_connected_components(self):
        """Benchmark strongly connected components on directed graph."""
        return list(nx.strongly_connected_components(self.directed_graph))

    def time_weakly_connected_components(self):
        """Benchmark weakly connected components on directed graph."""
        return list(nx.weakly_connected_components(self.directed_graph))

    def time_is_connected_small(self):
        """Benchmark connectivity check on small graph."""
        return nx.is_connected(self.small_graph)

    def time_is_connected_medium(self):
        """Benchmark connectivity check on medium graph."""
        return nx.is_connected(self.medium_graph)

    def time_number_connected_components_small(self):
        """Benchmark counting connected components on small graph."""
        return nx.number_connected_components(self.small_graph)

    def time_node_connectivity(self):
        """Benchmark node connectivity calculation."""
        if nx.is_connected(self.small_graph):
            return nx.node_connectivity(self.small_graph)
        return 0

    def time_edge_connectivity(self):
        """Benchmark edge connectivity calculation."""
        if nx.is_connected(self.small_graph):
            return nx.edge_connectivity(self.small_graph)
        return 0


class ClusteringSuite:
    """Benchmark suite for clustering algorithms."""

    def setup(self):
        """Set up test graphs for clustering benchmarks."""
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(300, 0.05, seed=42)
        self.triangle_graph = nx.erdos_renyi_graph(
            200, 0.2, seed=42
        )  # Higher edge probability

    def time_clustering_coefficient_small(self):
        """Benchmark clustering coefficient on small graph."""
        return nx.clustering(self.small_graph)

    def time_clustering_coefficient_medium(self):
        """Benchmark clustering coefficient on medium graph."""
        return nx.clustering(self.medium_graph)

    def time_average_clustering_small(self):
        """Benchmark average clustering on small graph."""
        return nx.average_clustering(self.small_graph)

    def time_average_clustering_medium(self):
        """Benchmark average clustering on medium graph."""
        return nx.average_clustering(self.medium_graph)

    def time_transitivity_small(self):
        """Benchmark transitivity on small graph."""
        return nx.transitivity(self.small_graph)

    def time_transitivity_medium(self):
        """Benchmark transitivity on medium graph."""
        return nx.transitivity(self.medium_graph)

    def time_triangles_small(self):
        """Benchmark triangle counting on small graph."""
        return nx.triangles(self.small_graph)

    def time_triangles_medium(self):
        """Benchmark triangle counting on medium graph."""
        return nx.triangles(self.medium_graph)

    def peakmem_clustering_large(self):
        """Benchmark peak memory for clustering on large graph."""
        large_graph = nx.erdos_renyi_graph(1000, 0.01, seed=42)
        return nx.clustering(large_graph)
