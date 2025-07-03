"""Algorithm service for graph analysis.

This module provides algorithms for graph analysis including pathfinding,
centrality measures, community detection, and other graph algorithms.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import networkx as nx

from ..caching.cache_service import CacheService
from ..core.base import Service
from ..services.graph_service import GraphService
from ..validators.algorithm_validator import AlgorithmValidator

logger = logging.getLogger(__name__)


class AlgorithmService(Service):
    """Service for graph algorithms."""

    def __init__(
        self,
        graph_service: GraphService,
        validator: AlgorithmValidator,
        cache_service: CacheService | None = None,
    ):
        super().__init__("AlgorithmService")
        self.graph_service = graph_service
        self.validator = validator
        self.cache_service = cache_service
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Algorithm registry
        self._algorithms = {
            # Path algorithms
            "shortest_path": self._shortest_path,
            "all_shortest_paths": self._all_shortest_paths,
            "shortest_path_length": self._shortest_path_length,
            # Centrality algorithms
            "degree_centrality": self._degree_centrality,
            "betweenness_centrality": self._betweenness_centrality,
            "closeness_centrality": self._closeness_centrality,
            "eigenvector_centrality": self._eigenvector_centrality,
            "pagerank": self._pagerank,
            # Community detection
            "connected_components": self._connected_components,
            "strongly_connected_components": self._strongly_connected_components,
            "weakly_connected_components": self._weakly_connected_components,
            # Clustering
            "clustering": self._clustering,
            "transitivity": self._transitivity,
            "triangle_count": self._triangle_count,
            # Structural measures
            "density": self._density,
            "diameter": self._diameter,
            "radius": self._radius,
            "center": self._center,
            "periphery": self._periphery,
            # Flow algorithms
            "maximum_flow": self._maximum_flow,
            "minimum_cut": self._minimum_cut,
            # Tree algorithms
            "minimum_spanning_tree": self._minimum_spanning_tree,
            "maximum_spanning_tree": self._maximum_spanning_tree,
        }

    async def initialize(self) -> None:
        """Initialize the algorithm service."""
        await super().initialize()
        logger.info("Algorithm service initialized")

    async def shutdown(self) -> None:
        """Shutdown the algorithm service."""
        await self._set_status(self.ComponentStatus.SHUTTING_DOWN)
        self._executor.shutdown(wait=True)
        await self._set_status(self.ComponentStatus.SHUTDOWN)
        logger.info("Algorithm service shutdown")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self.status == self.ComponentStatus.READY,
            "available_algorithms": len(self._algorithms),
            "executor_active": not self._executor._shutdown,
        }

    async def run_algorithm(
        self, algorithm: str, graph_id: str, **parameters
    ) -> dict[str, Any]:
        """Run a graph algorithm."""
        async with self.track_request():
            # Validate algorithm request
            validation = await self.validator.validate_algorithm_request(
                {"algorithm": algorithm, "graph_id": graph_id, "parameters": parameters}
            )

            if not validation.valid:
                raise ValueError(f"Invalid algorithm request: {validation.errors}")

            # Check cache first
            cache_key = self._build_cache_key(algorithm, graph_id, parameters)
            if self.cache_service:
                cached_result = await self.cache_service.get(cache_key)
                if cached_result:
                    logger.debug(
                        f"Returning cached result for {algorithm} on {graph_id}"
                    )
                    return cached_result

            # Get the algorithm function
            if algorithm not in self._algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            algorithm_func = self._algorithms[algorithm]

            # Get the graph
            graph = await self.graph_service.get_graph(graph_id)

            # Run the algorithm
            start_time = time.time()
            try:
                result = await self._run_algorithm_safe(
                    algorithm_func, graph, **parameters
                )
                execution_time = time.time() - start_time

                response = {
                    "algorithm": algorithm,
                    "graph_id": graph_id,
                    "result": result,
                    "execution_time": execution_time,
                    "parameters": parameters,
                }

                # Cache the result
                if self.cache_service:
                    ttl = self._get_cache_ttl(algorithm, graph.number_of_nodes())
                    await self.cache_service.set(cache_key, response, ttl=ttl)

                logger.debug(
                    f"Algorithm {algorithm} completed in {execution_time:.3f}s"
                )
                return response

            except Exception as e:
                logger.error(f"Algorithm {algorithm} failed: {e}")
                raise

    async def _run_algorithm_safe(
        self, algorithm_func: Callable, graph: nx.Graph, **parameters
    ) -> Any:
        """Run algorithm in thread pool for CPU-intensive operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, lambda: algorithm_func(graph, **parameters)
        )

    # Path algorithms
    def _shortest_path(
        self,
        graph: nx.Graph,
        source: str,
        target: str,
        weight: str | None = None,
        method: str = "dijkstra",
    ) -> dict[str, Any]:
        """Calculate shortest path between two nodes."""
        try:
            if method == "dijkstra" and weight:
                path = nx.dijkstra_path(graph, source, target, weight=weight)
                length = nx.dijkstra_path_length(graph, source, target, weight=weight)
            elif method == "bellman_ford" and weight:
                path = nx.bellman_ford_path(graph, source, target, weight=weight)
                length = nx.bellman_ford_path_length(
                    graph, source, target, weight=weight
                )
            else:
                path = nx.shortest_path(graph, source, target, weight=weight)
                length = nx.shortest_path_length(graph, source, target, weight=weight)

            return {
                "path": path,
                "length": length,
                "method": method,
                "weighted": weight is not None,
            }
        except nx.NetworkXNoPath:
            return {
                "path": None,
                "length": float("inf"),
                "method": method,
                "weighted": weight is not None,
                "error": "No path exists",
            }

    def _all_shortest_paths(
        self, graph: nx.Graph, source: str, target: str, weight: str | None = None
    ) -> dict[str, Any]:
        """Find all shortest paths between two nodes."""
        try:
            paths = list(nx.all_shortest_paths(graph, source, target, weight=weight))
            length = nx.shortest_path_length(graph, source, target, weight=weight)

            return {
                "paths": paths,
                "path_count": len(paths),
                "length": length,
                "weighted": weight is not None,
            }
        except nx.NetworkXNoPath:
            return {
                "paths": [],
                "path_count": 0,
                "length": float("inf"),
                "weighted": weight is not None,
                "error": "No path exists",
            }

    def _shortest_path_length(
        self,
        graph: nx.Graph,
        source: str | None = None,
        target: str | None = None,
        weight: str | None = None,
    ) -> dict[str, Any]:
        """Calculate shortest path lengths."""
        if source and target:
            # Single pair
            try:
                length = nx.shortest_path_length(graph, source, target, weight=weight)
                return {"length": length, "source": source, "target": target}
            except nx.NetworkXNoPath:
                return {
                    "length": float("inf"),
                    "source": source,
                    "target": target,
                    "error": "No path",
                }
        elif source:
            # From single source to all nodes
            lengths = dict(
                nx.single_source_shortest_path_length(graph, source, weight=weight)
            )
            return {"lengths": lengths, "source": source}
        else:
            # All pairs (expensive for large graphs)
            if graph.number_of_nodes() > 1000:
                raise ValueError(
                    "All-pairs shortest path too expensive for large graphs"
                )

            lengths = dict(nx.all_pairs_shortest_path_length(graph, weight=weight))
            return {"lengths": lengths}

    # Centrality algorithms
    def _degree_centrality(self, graph: nx.Graph) -> dict[str, float]:
        """Calculate degree centrality."""
        return nx.degree_centrality(graph)

    def _betweenness_centrality(
        self,
        graph: nx.Graph,
        k: int | None = None,
        normalized: bool = True,
        weight: str | None = None,
    ) -> dict[str, float]:
        """Calculate betweenness centrality."""
        return nx.betweenness_centrality(
            graph, k=k, normalized=normalized, weight=weight
        )

    def _closeness_centrality(
        self, graph: nx.Graph, distance: str | None = None, wf_improved: bool = True
    ) -> dict[str, float]:
        """Calculate closeness centrality."""
        return nx.closeness_centrality(
            graph, distance=distance, wf_improved=wf_improved
        )

    def _eigenvector_centrality(
        self,
        graph: nx.Graph,
        max_iter: int = 100,
        tol: float = 1e-06,
        weight: str | None = None,
    ) -> dict[str, float]:
        """Calculate eigenvector centrality."""
        try:
            return nx.eigenvector_centrality(
                graph, max_iter=max_iter, tol=tol, weight=weight
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback to numpy method
            return nx.eigenvector_centrality_numpy(graph, weight=weight)

    def _pagerank(
        self,
        graph: nx.Graph,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-06,
        weight: str | None = None,
        dangling: dict | None = None,
    ) -> dict[str, float]:
        """Calculate PageRank."""
        return nx.pagerank(
            graph,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            weight=weight,
            dangling=dangling,
        )

    # Community detection
    def _connected_components(self, graph: nx.Graph) -> dict[str, Any]:
        """Find connected components."""
        if graph.is_directed():
            components = list(nx.weakly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))

        return {
            "components": [list(component) for component in components],
            "component_count": len(components),
            "largest_component_size": (
                max(len(comp) for comp in components) if components else 0
            ),
        }

    def _strongly_connected_components(self, graph: nx.Graph) -> dict[str, Any]:
        """Find strongly connected components (directed graphs only)."""
        if not graph.is_directed():
            raise ValueError(
                "Strongly connected components only defined for directed graphs"
            )

        components = list(nx.strongly_connected_components(graph))
        return {
            "components": [list(component) for component in components],
            "component_count": len(components),
            "largest_component_size": (
                max(len(comp) for comp in components) if components else 0
            ),
        }

    def _weakly_connected_components(self, graph: nx.Graph) -> dict[str, Any]:
        """Find weakly connected components (directed graphs only)."""
        if not graph.is_directed():
            raise ValueError(
                "Weakly connected components only defined for directed graphs"
            )

        components = list(nx.weakly_connected_components(graph))
        return {
            "components": [list(component) for component in components],
            "component_count": len(components),
            "largest_component_size": (
                max(len(comp) for comp in components) if components else 0
            ),
        }

    # Clustering algorithms
    def _clustering(
        self, graph: nx.Graph, nodes: list[str] | None = None, weight: str | None = None
    ) -> dict[str, Any]:
        """Calculate clustering coefficients."""
        if nodes:
            clustering = {
                node: nx.clustering(graph, node, weight=weight) for node in nodes
            }
        else:
            clustering = nx.clustering(graph, weight=weight)

        if isinstance(clustering, dict):
            avg_clustering = (
                sum(clustering.values()) / len(clustering) if clustering else 0
            )
        else:
            avg_clustering = clustering

        return {"clustering": clustering, "average_clustering": avg_clustering}

    def _transitivity(self, graph: nx.Graph) -> float:
        """Calculate graph transitivity."""
        return nx.transitivity(graph)

    def _triangle_count(
        self, graph: nx.Graph, nodes: list[str] | None = None
    ) -> dict[str, Any]:
        """Count triangles in the graph."""
        if nodes:
            triangles = {node: nx.triangles(graph, node) for node in nodes}
        else:
            triangles = nx.triangles(graph)

        total_triangles = sum(triangles.values()) // 3  # Each triangle counted 3 times

        return {"triangles": triangles, "total_triangles": total_triangles}

    # Structural measures
    def _density(self, graph: nx.Graph) -> float:
        """Calculate graph density."""
        return nx.density(graph)

    def _diameter(self, graph: nx.Graph) -> int:
        """Calculate graph diameter."""
        if not nx.is_connected(graph):
            raise ValueError("Graph must be connected to calculate diameter")
        return nx.diameter(graph)

    def _radius(self, graph: nx.Graph) -> int:
        """Calculate graph radius."""
        if not nx.is_connected(graph):
            raise ValueError("Graph must be connected to calculate radius")
        return nx.radius(graph)

    def _center(self, graph: nx.Graph) -> list[str]:
        """Find graph center nodes."""
        if not nx.is_connected(graph):
            raise ValueError("Graph must be connected to find center")
        return list(nx.center(graph))

    def _periphery(self, graph: nx.Graph) -> list[str]:
        """Find graph periphery nodes."""
        if not nx.is_connected(graph):
            raise ValueError("Graph must be connected to find periphery")
        return list(nx.periphery(graph))

    # Flow algorithms
    def _maximum_flow(
        self, graph: nx.Graph, source: str, target: str, capacity: str = "capacity"
    ) -> dict[str, Any]:
        """Calculate maximum flow."""
        flow_value, flow_dict = nx.maximum_flow(
            graph, source, target, capacity=capacity
        )
        return {"max_flow_value": flow_value, "flow_dict": flow_dict}

    def _minimum_cut(
        self, graph: nx.Graph, source: str, target: str, capacity: str = "capacity"
    ) -> dict[str, Any]:
        """Calculate minimum cut."""
        cut_value, partition = nx.minimum_cut(graph, source, target, capacity=capacity)
        return {
            "cut_value": cut_value,
            "partition": [list(partition[0]), list(partition[1])],
        }

    # Tree algorithms
    def _minimum_spanning_tree(
        self, graph: nx.Graph, weight: str = "weight", algorithm: str = "kruskal"
    ) -> dict[str, Any]:
        """Calculate minimum spanning tree."""
        if algorithm == "kruskal":
            mst = nx.minimum_spanning_tree(graph, weight=weight, algorithm="kruskal")
        elif algorithm == "prim":
            mst = nx.minimum_spanning_tree(graph, weight=weight, algorithm="prim")
        else:
            mst = nx.minimum_spanning_tree(graph, weight=weight)

        total_weight = sum(data.get(weight, 1) for _, _, data in mst.edges(data=True))

        return {
            "edges": [(u, v, data) for u, v, data in mst.edges(data=True)],
            "total_weight": total_weight,
            "algorithm": algorithm,
        }

    def _maximum_spanning_tree(
        self, graph: nx.Graph, weight: str = "weight", algorithm: str = "kruskal"
    ) -> dict[str, Any]:
        """Calculate maximum spanning tree."""
        if algorithm == "kruskal":
            mst = nx.maximum_spanning_tree(graph, weight=weight, algorithm="kruskal")
        elif algorithm == "prim":
            mst = nx.maximum_spanning_tree(graph, weight=weight, algorithm="prim")
        else:
            mst = nx.maximum_spanning_tree(graph, weight=weight)

        total_weight = sum(data.get(weight, 1) for _, _, data in mst.edges(data=True))

        return {
            "edges": [(u, v, data) for u, v, data in mst.edges(data=True)],
            "total_weight": total_weight,
            "algorithm": algorithm,
        }

    def _build_cache_key(
        self, algorithm: str, graph_id: str, parameters: dict[str, Any]
    ) -> str:
        """Build cache key for algorithm result."""
        param_str = "_".join(f"{k}:{v}" for k, v in sorted(parameters.items()))
        return f"algorithm:{algorithm}:{graph_id}:{hash(param_str) % 10000}"

    def _get_cache_ttl(self, algorithm: str, graph_size: int) -> int:
        """Get cache TTL based on algorithm and graph size."""
        # Expensive algorithms get longer cache TTL
        expensive_algorithms = {
            "betweenness_centrality",
            "all_shortest_paths",
            "eigenvector_centrality",
            "strongly_connected_components",
            "minimum_spanning_tree",
        }

        base_ttl = (
            3600 if algorithm in expensive_algorithms else 1800
        )  # 1 hour vs 30 min

        # Larger graphs get longer cache TTL
        if graph_size > 10000:
            return base_ttl * 2
        elif graph_size > 1000:
            return base_ttl
        else:
            return base_ttl // 2

    def get_available_algorithms(self) -> list[str]:
        """Get list of available algorithms."""
        return list(self._algorithms.keys())
