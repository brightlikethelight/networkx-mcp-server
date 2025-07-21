"""Benchmarks for MCP handler operations.

This module benchmarks the performance of the Model Context Protocol
handlers used in the NetworkX MCP server.
"""

import asyncio
from unittest.mock import Mock

import networkx as nx

from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.mcp.handlers.algorithms import AlgorithmHandler
from networkx_mcp.mcp.handlers.analysis import AnalysisHandler
from networkx_mcp.mcp.handlers.graph_ops import GraphOpsHandler


class MCPHandlersSuite:
    """Benchmark suite for MCP handler operations."""

    def setup(self):
        """Set up MCP handlers and test data."""
        # Mock MCP server
        self.mock_mcp = Mock()
        self.mock_mcp.tool = Mock(return_value=lambda func: func)

        # Graph manager
        self.graph_manager = GraphManager()

        # Create test graphs
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(300, 0.05, seed=42)
        self.scale_free = nx.barabasi_albert_graph(200, 3, seed=42)

        # Add weights for algorithm testing
        for u, v in self.small_graph.edges():
            self.small_graph.edges[u, v]["weight"] = 1.0

        # Add graphs to manager
        self.graph_manager.create_graph("small")
        self.graph_manager.graphs["small"] = self.small_graph

        self.graph_manager.create_graph("medium")
        self.graph_manager.graphs["medium"] = self.medium_graph

        self.graph_manager.create_graph("scale_free")
        self.graph_manager.graphs["scale_free"] = self.scale_free

        # Initialize handlers
        self.graph_ops_handler = GraphOpsHandler(self.mock_mcp, self.graph_manager)
        self.algorithm_handler = AlgorithmHandler(self.mock_mcp, self.graph_manager)
        self.analysis_handler = AnalysisHandler(self.mock_mcp, self.graph_manager)

    def time_graph_info_retrieval_small(self):
        """Benchmark graph info retrieval for small graph."""
        return self.graph_manager.get_graph_info("small")

    def time_graph_info_retrieval_medium(self):
        """Benchmark graph info retrieval for medium graph."""
        return self.graph_manager.get_graph_info("medium")

    def time_list_graphs(self):
        """Benchmark listing all graphs."""
        return self.graph_manager.list_graphs()

    def time_graph_statistics_small(self):
        """Benchmark graph statistics calculation for small graph."""
        # Simulate the graph_statistics tool
        G = self.graph_manager.get_graph("small")
        if G:
            stats = {
                "basic": {
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "is_directed": G.is_directed(),
                    "is_multigraph": G.is_multigraph(),
                }
            }

            # Degree statistics
            degrees = [d for n, d in G.degree()]
            if degrees:
                stats["degree"] = {
                    "average": sum(degrees) / len(degrees),
                    "max": max(degrees),
                    "min": min(degrees),
                }

            # Connectivity
            stats["connectivity"] = {
                "is_connected": nx.is_connected(G),
                "num_connected_components": nx.number_connected_components(G),
            }

            # Clustering
            stats["clustering"] = {
                "average_clustering": nx.average_clustering(G),
                "transitivity": nx.transitivity(G),
            }

            return stats

    def time_graph_statistics_medium(self):
        """Benchmark graph statistics calculation for medium graph."""
        G = self.graph_manager.get_graph("medium")
        if G:
            stats = {
                "basic": {
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "density": nx.density(G),
                }
            }

            # Only compute expensive operations for benchmarking
            stats["connectivity"] = {
                "is_connected": nx.is_connected(G),
                "num_connected_components": nx.number_connected_components(G),
            }

            return stats

    def time_shortest_path_handler_small(self):
        """Benchmark shortest path through handler interface."""
        G = self.graph_manager.get_graph("small")
        if G and G.number_of_nodes() >= 2:
            nodes = list(G.nodes())
            source, target = nodes[0], nodes[-1]

            try:
                path = nx.shortest_path(G, source, target, weight="weight")
                length = nx.shortest_path_length(G, source, target, weight="weight")
                return {
                    "path": path,
                    "length": length,
                    "source": source,
                    "target": target,
                }
            except nx.NetworkXNoPath:
                return {"error": "No path found"}

    def time_centrality_calculation_small(self):
        """Benchmark centrality calculation through handler interface."""
        G = self.graph_manager.get_graph("small")
        if G:
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

            values = list(centrality.values())
            stats = {
                "mean": sum(values) / len(values) if values else 0,
                "max": max(values) if values else 0,
                "min": min(values) if values else 0,
            }

            return {
                "centrality_type": "degree",
                "top_nodes": sorted_nodes[:10],
                "statistics": stats,
            }

    def time_community_detection_small(self):
        """Benchmark community detection through handler interface."""
        G = self.graph_manager.get_graph("small")
        if G and G.number_of_edges() > 0:
            communities = list(nx.community.greedy_modularity_communities(G))
            communities = [list(c) for c in communities]
            modularity = nx.community.modularity(G, communities)

            return {
                "method": "greedy_modularity",
                "num_communities": len(communities),
                "modularity": float(modularity),
                "communities": communities[:10],  # First 10 communities
                "community_sizes": [len(c) for c in communities],
            }

    def time_bipartite_analysis_small(self):
        """Benchmark bipartite analysis."""
        G = self.graph_manager.get_graph("small")
        if G:
            from networkx.algorithms import bipartite

            is_bipartite_result = bipartite.is_bipartite(G)

            if is_bipartite_result:
                node_sets = bipartite.sets(G)
                set1, set2 = list(node_sets[0]), list(node_sets[1])
                density = bipartite.density(G, set1)

                return {
                    "is_bipartite": True,
                    "set1_size": len(set1),
                    "set2_size": len(set2),
                    "density": density,
                }
            else:
                return {
                    "is_bipartite": False,
                    "reason": "Graph contains odd-length cycles",
                }

    def peakmem_large_graph_operations(self):
        """Benchmark peak memory for large graph operations."""
        # Create a large graph temporarily
        large_graph = nx.erdos_renyi_graph(1000, 0.02, seed=42)

        self.graph_manager.create_graph("large_temp")
        self.graph_manager.graphs["large_temp"] = large_graph

        # Perform memory-intensive operations
        self.graph_manager.get_graph_info("large_temp")
        stats = {
            "num_nodes": large_graph.number_of_nodes(),
            "num_edges": large_graph.number_of_edges(),
            "density": nx.density(large_graph),
            "is_connected": nx.is_connected(large_graph),
        }

        # Cleanup
        self.graph_manager.delete_graph("large_temp")

        return stats


class AsyncMCPHandlersSuite:
    """Benchmark suite for async MCP handler operations."""

    def setup(self):
        """Set up async MCP handlers and test data."""
        # Mock MCP server
        self.mock_mcp = Mock()
        self.mock_mcp.tool = Mock(return_value=lambda func: func)

        # Graph manager
        self.graph_manager = GraphManager()

        # Create test graphs
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)

        # Add to manager
        self.graph_manager.create_graph("async_small")
        self.graph_manager.graphs["async_small"] = self.small_graph

        # Initialize handlers
        self.graph_ops_handler = GraphOpsHandler(self.mock_mcp, self.graph_manager)
        self.algorithm_handler = AlgorithmHandler(self.mock_mcp, self.graph_manager)

    def time_async_graph_retrieval(self):
        """Benchmark async graph retrieval."""

        async def async_operation():
            return self.graph_manager.get_graph("async_small")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_operation())
        finally:
            loop.close()

    def time_async_statistics_calculation(self):
        """Benchmark async statistics calculation."""

        async def async_stats():
            G = self.graph_manager.get_graph("async_small")
            if G:
                return {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "clustering": nx.average_clustering(G),
                }
            return {}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_stats())
        finally:
            loop.close()

    def peakmem_async_operations(self):
        """Benchmark peak memory for async operations."""

        async def memory_intensive_async():
            G = self.graph_manager.get_graph("async_small")
            if G:
                # Multiple concurrent operations
                results = await asyncio.gather(
                    *[self._simulate_async_operation(G) for _ in range(5)]
                )
                return results
            return []

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(memory_intensive_async())
        finally:
            loop.close()

    async def _simulate_async_operation(self, graph):
        """Simulate an async graph operation."""
        return {
            "centrality": nx.degree_centrality(graph),
            "components": list(nx.connected_components(graph)),
        }
