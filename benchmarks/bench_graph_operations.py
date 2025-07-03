"""Benchmarks for core graph operations.

This module benchmarks the performance of fundamental graph operations
used throughout the NetworkX MCP server.
"""

import networkx as nx
import numpy as np

from networkx_mcp.core.graph_operations import GraphManager


class GraphOperationsSuite:
    """Benchmark suite for graph operations."""

    def setup(self):
        """Set up test data for benchmarks."""
        self.graph_manager = GraphManager()

        # Create test graphs of various sizes
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(500, 0.05, seed=42)
        self.large_graph = nx.erdos_renyi_graph(1000, 0.02, seed=42)

        # Scale-free graph
        self.scale_free = nx.barabasi_albert_graph(500, 3, seed=42)

        # Small-world graph
        self.small_world = nx.watts_strogatz_graph(500, 6, 0.3, seed=42)

        # Dense graph
        self.dense_graph = nx.erdos_renyi_graph(200, 0.3, seed=42)

        # Add graphs to manager
        self.graph_manager.create_graph("small")
        self.graph_manager.graphs["small"] = self.small_graph

        self.graph_manager.create_graph("medium")
        self.graph_manager.graphs["medium"] = self.medium_graph

        self.graph_manager.create_graph("large")
        self.graph_manager.graphs["large"] = self.large_graph

    def time_graph_creation_small(self):
        """Benchmark small graph creation."""
        self.graph_manager.create_graph("test_small")
        G = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.graph_manager.graphs["test_small"] = G
        self.graph_manager.delete_graph("test_small")

    def time_graph_creation_medium(self):
        """Benchmark medium graph creation."""
        self.graph_manager.create_graph("test_medium")
        G = nx.erdos_renyi_graph(500, 0.05, seed=42)
        self.graph_manager.graphs["test_medium"] = G
        self.graph_manager.delete_graph("test_medium")

    def time_graph_creation_large(self):
        """Benchmark large graph creation."""
        self.graph_manager.create_graph("test_large")
        G = nx.erdos_renyi_graph(1000, 0.02, seed=42)
        self.graph_manager.graphs["test_large"] = G
        self.graph_manager.delete_graph("test_large")

    def time_graph_retrieval_small(self):
        """Benchmark small graph retrieval."""
        return self.graph_manager.get_graph("small")

    def time_graph_retrieval_medium(self):
        """Benchmark medium graph retrieval."""
        return self.graph_manager.get_graph("medium")

    def time_graph_retrieval_large(self):
        """Benchmark large graph retrieval."""
        return self.graph_manager.get_graph("large")

    def time_node_addition_batch_small(self):
        """Benchmark batch node addition to small graph."""
        G = self.small_graph.copy()
        nodes = [f"new_node_{i}" for i in range(50)]
        G.add_nodes_from(nodes)

    def time_node_addition_batch_medium(self):
        """Benchmark batch node addition to medium graph."""
        G = self.medium_graph.copy()
        nodes = [f"new_node_{i}" for i in range(100)]
        G.add_nodes_from(nodes)

    def time_edge_addition_batch_small(self):
        """Benchmark batch edge addition to small graph."""
        G = self.small_graph.copy()
        nodes = list(G.nodes())
        if len(nodes) >= 2:
            edges = [
                (nodes[i], nodes[j])
                for i in range(0, min(50, len(nodes)), 2)
                for j in range(1, min(51, len(nodes)), 2)
                if i != j
            ]
            G.add_edges_from(edges[:25])

    def time_edge_addition_batch_medium(self):
        """Benchmark batch edge addition to medium graph."""
        G = self.medium_graph.copy()
        nodes = list(G.nodes())
        if len(nodes) >= 2:
            edges = [
                (nodes[i], nodes[j])
                for i in range(0, min(100, len(nodes)), 2)
                for j in range(1, min(101, len(nodes)), 2)
                if i != j
            ]
            G.add_edges_from(edges[:50])

    def time_subgraph_extraction_small(self):
        """Benchmark subgraph extraction from small graph."""
        nodes = list(self.small_graph.nodes())[:50]
        return self.small_graph.subgraph(nodes)

    def time_subgraph_extraction_medium(self):
        """Benchmark subgraph extraction from medium graph."""
        nodes = list(self.medium_graph.nodes())[:100]
        return self.medium_graph.subgraph(nodes)

    def peakmem_graph_creation_large(self):
        """Benchmark peak memory usage for large graph creation."""
        self.graph_manager.create_graph("memory_test")
        G = nx.erdos_renyi_graph(2000, 0.01, seed=42)
        self.graph_manager.graphs["memory_test"] = G
        self.graph_manager.delete_graph("memory_test")

    def peakmem_dense_graph_operations(self):
        """Benchmark peak memory usage for dense graph operations."""
        G = nx.erdos_renyi_graph(300, 0.5, seed=42)
        # Perform memory-intensive operations
        nodes = list(G.nodes())
        edges = list(G.edges())
        subgraph = G.subgraph(nodes[:100])
        return subgraph


class GraphGenerationSuite:
    """Benchmark suite for graph generation algorithms."""

    params = [100, 500, 1000]
    param_names = ["n_nodes"]

    def time_erdos_renyi_generation(self, n_nodes):
        """Benchmark Erdős-Rényi graph generation."""
        return nx.erdos_renyi_graph(n_nodes, 0.1, seed=42)

    def time_barabasi_albert_generation(self, n_nodes):
        """Benchmark Barabási-Albert graph generation."""
        m = min(5, n_nodes // 10)
        return nx.barabasi_albert_graph(n_nodes, m, seed=42)

    def time_watts_strogatz_generation(self, n_nodes):
        """Benchmark Watts-Strogatz graph generation."""
        k = min(6, n_nodes // 10)
        return nx.watts_strogatz_graph(n_nodes, k, 0.3, seed=42)

    def time_complete_graph_generation(self, n_nodes):
        """Benchmark complete graph generation."""
        if n_nodes <= 500:  # Avoid memory issues
            return nx.complete_graph(n_nodes)

    def peakmem_graph_generation(self, n_nodes):
        """Benchmark peak memory for graph generation."""
        return nx.erdos_renyi_graph(n_nodes, 0.1, seed=42)


class GraphConversionSuite:
    """Benchmark suite for graph format conversions."""

    def setup(self):
        """Set up test graphs for conversion benchmarks."""
        self.graph_small = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.graph_medium = nx.erdos_renyi_graph(500, 0.05, seed=42)

        # Add attributes for more realistic conversion scenarios
        for node in self.graph_small.nodes():
            self.graph_small.nodes[node]["weight"] = np.random.random()
            self.graph_small.nodes[node]["category"] = f"cat_{node % 5}"

        for u, v in self.graph_small.edges():
            self.graph_small.edges[u, v]["weight"] = np.random.random()

    def time_to_dict_small(self):
        """Benchmark conversion to dictionary format."""
        return nx.to_dict_of_dicts(self.graph_small)

    def time_to_dict_medium(self):
        """Benchmark conversion to dictionary format."""
        return nx.to_dict_of_dicts(self.graph_medium)

    def time_to_numpy_small(self):
        """Benchmark conversion to NumPy array."""
        return nx.to_numpy_array(self.graph_small)

    def time_to_numpy_medium(self):
        """Benchmark conversion to NumPy array."""
        return nx.to_numpy_array(self.graph_medium)

    def time_from_numpy_small(self):
        """Benchmark creation from NumPy array."""
        adj_matrix = nx.to_numpy_array(self.graph_small)
        return nx.from_numpy_array(adj_matrix)

    def time_json_serialization_small(self):
        """Benchmark JSON serialization."""
        from networkx.readwrite import json_graph

        return json_graph.node_link_data(self.graph_small)

    def time_json_deserialization_small(self):
        """Benchmark JSON deserialization."""
        from networkx.readwrite import json_graph

        data = json_graph.node_link_data(self.graph_small)
        return json_graph.node_link_graph(data)

    def peakmem_large_conversion(self):
        """Benchmark peak memory for large graph conversion."""
        large_graph = nx.erdos_renyi_graph(1000, 0.02, seed=42)
        return nx.to_numpy_array(large_graph)
