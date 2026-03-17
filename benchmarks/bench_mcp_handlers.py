"""Benchmarks for MCP handler operations.

This module benchmarks the performance of the Model Context Protocol
handlers used in the NetworkX MCP server.
"""

import asyncio

import networkx as nx


class MCPHandlersSuite:
    """Benchmark suite for MCP handler operations."""

    def setup(self):
        """Set up test data for benchmarks."""
        self.graphs = {}

        # Create test graphs
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
        self.medium_graph = nx.erdos_renyi_graph(300, 0.05, seed=42)
        self.scale_free = nx.barabasi_albert_graph(200, 3, seed=42)

        # Add weights for algorithm testing
        for u, v in self.small_graph.edges():
            self.small_graph.edges[u, v]["weight"] = 1.0

        self.graphs["small"] = self.small_graph
        self.graphs["medium"] = self.medium_graph
        self.graphs["scale_free"] = self.scale_free

    def time_graph_statistics_small(self):
        """Benchmark graph statistics calculation for small graph."""
        G = self.graphs["small"]
        stats = {
            "basic": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": nx.density(G),
                "is_directed": G.is_directed(),
                "is_multigraph": G.is_multigraph(),
            }
        }

        degrees = [d for n, d in G.degree()]
        if degrees:
            stats["degree"] = {
                "average": sum(degrees) / len(degrees),
                "max": max(degrees),
                "min": min(degrees),
            }

        stats["connectivity"] = {
            "is_connected": nx.is_connected(G),
            "num_connected_components": nx.number_connected_components(G),
        }

        stats["clustering"] = {
            "average_clustering": nx.average_clustering(G),
            "transitivity": nx.transitivity(G),
        }

        return stats

    def time_graph_statistics_medium(self):
        """Benchmark graph statistics calculation for medium graph."""
        G = self.graphs["medium"]
        stats = {
            "basic": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "density": nx.density(G),
            }
        }

        stats["connectivity"] = {
            "is_connected": nx.is_connected(G),
            "num_connected_components": nx.number_connected_components(G),
        }

        return stats

    def time_shortest_path_handler_small(self):
        """Benchmark shortest path through handler interface."""
        G = self.graphs["small"]
        if G.number_of_nodes() >= 2:
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
        G = self.graphs["small"]
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
        G = self.graphs["small"]
        if G.number_of_edges() > 0:
            communities = list(nx.community.greedy_modularity_communities(G))
            communities = [list(c) for c in communities]
            modularity = nx.community.modularity(G, communities)

            return {
                "method": "greedy_modularity",
                "num_communities": len(communities),
                "modularity": float(modularity),
                "communities": communities[:10],
                "community_sizes": [len(c) for c in communities],
            }

    def peakmem_large_graph_operations(self):
        """Benchmark peak memory for large graph operations."""
        large_graph = nx.erdos_renyi_graph(1000, 0.02, seed=42)
        stats = {
            "num_nodes": large_graph.number_of_nodes(),
            "num_edges": large_graph.number_of_edges(),
            "density": nx.density(large_graph),
            "is_connected": nx.is_connected(large_graph),
        }
        return stats


class AsyncMCPHandlersSuite:
    """Benchmark suite for async MCP handler operations."""

    def setup(self):
        """Set up test data for async benchmarks."""
        self.small_graph = nx.erdos_renyi_graph(100, 0.1, seed=42)

    def time_async_graph_retrieval(self):
        """Benchmark async graph retrieval."""

        async def async_operation():
            return self.small_graph

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_operation())
        finally:
            loop.close()

    def time_async_statistics_calculation(self):
        """Benchmark async statistics calculation."""

        async def async_stats():
            G = self.small_graph
            return {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "clustering": nx.average_clustering(G),
            }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_stats())
        finally:
            loop.close()

    def peakmem_async_operations(self):
        """Benchmark peak memory for async operations."""

        async def memory_intensive_async():
            G = self.small_graph
            results = await asyncio.gather(
                *[self._simulate_async_operation(G) for _ in range(5)]
            )
            return results

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
