"""Algorithm Handler v2 for NetworkX MCP Server.

This module provides the modernized MCP algorithm handlers that integrate
with the new service-oriented architecture using dependency injection.
"""

import logging
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from ...core.container import Container
from ...services.algorithm_service import AlgorithmService
from ...validators.algorithm_validator import AlgorithmValidator

logger = logging.getLogger(__name__)


class AlgorithmHandler:
    """Modern handler for graph algorithm operations using service architecture."""

    def __init__(self, mcp: FastMCP, container: Container):
        """Initialize the handler with MCP server and DI container."""
        self.mcp = mcp
        self.container = container
        self._algorithm_service: Optional[AlgorithmService] = None
        self._validator: Optional[AlgorithmValidator] = None

    async def initialize(self) -> None:
        """Initialize the handler and resolve dependencies."""
        self._algorithm_service = await self.container.resolve(AlgorithmService)
        self._validator = await self.container.resolve(AlgorithmValidator)
        self._register_tools()
        logger.info("Algorithm handler initialized")

    def _register_tools(self):
        """Register all algorithm tools."""

        @self.mcp.tool()
        async def run_algorithm(
            algorithm: str, graph_id: str, **parameters
        ) -> Dict[str, Any]:
            """Run a graph algorithm.

            Args:
                algorithm: Name of the algorithm to run
                graph_id: ID of the graph to analyze
                **parameters: Algorithm-specific parameters

            Returns:
                Dict with algorithm results
            """
            try:
                result = await self._algorithm_service.run_algorithm(
                    algorithm=algorithm, graph_id=graph_id, **parameters
                )
                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(
                    f"Failed to run algorithm {algorithm} on graph {graph_id}: {e}"
                )
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_supported_algorithms() -> Dict[str, Any]:
            """Get list of supported algorithms.

            Returns:
                Dict with supported algorithms and their descriptions
            """
            try:
                algorithms = self._algorithm_service.get_available_algorithms()

                # Get detailed info for each algorithm
                algorithm_info = {}
                for algorithm in algorithms:
                    info = self._validator.get_algorithm_info(algorithm)
                    if info:
                        algorithm_info[algorithm] = info

                return {
                    "algorithms": algorithms,
                    "count": len(algorithms),
                    "details": algorithm_info,
                }
            except Exception as e:
                logger.error(f"Failed to get supported algorithms: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
            """Get detailed information about a specific algorithm.

            Args:
                algorithm: Name of the algorithm

            Returns:
                Dict with algorithm details
            """
            try:
                info = self._validator.get_algorithm_info(algorithm)
                if info is None:
                    return {"error": f"Algorithm '{algorithm}' not found"}

                return {"algorithm": algorithm, **info}
            except Exception as e:
                logger.error(f"Failed to get algorithm info for {algorithm}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def validate_algorithm_request(
            algorithm: str, graph_id: str, **parameters
        ) -> Dict[str, Any]:
            """Validate an algorithm request before execution.

            Args:
                algorithm: Name of the algorithm
                graph_id: ID of the graph
                **parameters: Algorithm parameters

            Returns:
                Dict with validation results
            """
            try:
                # Validate the request
                validation = await self._validator.validate_algorithm_request(
                    {
                        "algorithm": algorithm,
                        "graph_id": graph_id,
                        "parameters": parameters,
                    }
                )

                result = {
                    "algorithm": algorithm,
                    "graph_id": graph_id,
                    "valid": validation.valid,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                }

                # If basic validation passes, check compatibility
                if validation.valid:
                    from ...services.graph_service import GraphService

                    graph_service = await self.container.resolve(GraphService)
                    graph = await graph_service.get_graph(graph_id)

                    compatibility = (
                        await self._validator.validate_algorithm_compatibility(
                            algorithm, graph
                        )
                    )

                    result["compatibility"] = {
                        "valid": compatibility.valid,
                        "errors": compatibility.errors,
                        "warnings": compatibility.warnings,
                    }

                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to validate algorithm request: {e}")
                return {"error": f"Internal error: {str(e)}"}

        # Convenience methods for common algorithms
        @self.mcp.tool()
        async def shortest_path(
            graph_id: str,
            source: str,
            target: str,
            weight: Optional[str] = None,
            method: str = "dijkstra",
        ) -> Dict[str, Any]:
            """Find shortest path between two nodes.

            Args:
                graph_id: ID of the graph
                source: Source node
                target: Target node
                weight: Edge attribute to use as weight (optional)
                method: Algorithm to use ('dijkstra', 'bellman_ford')

            Returns:
                Dict with path information
            """
            return await run_algorithm(
                algorithm="shortest_path",
                graph_id=graph_id,
                source=source,
                target=target,
                weight=weight,
                method=method,
            )

        @self.mcp.tool()
        async def centrality_analysis(
            graph_id: str, centrality_type: str = "degree", **parameters
        ) -> Dict[str, Any]:
            """Perform centrality analysis on a graph.

            Args:
                graph_id: ID of the graph
                centrality_type: Type of centrality ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank')
                **parameters: Centrality-specific parameters

            Returns:
                Dict with centrality values
            """
            algorithm_map = {
                "degree": "degree_centrality",
                "betweenness": "betweenness_centrality",
                "closeness": "closeness_centrality",
                "eigenvector": "eigenvector_centrality",
                "pagerank": "pagerank",
            }

            if centrality_type not in algorithm_map:
                return {"error": f"Unsupported centrality type: {centrality_type}"}

            return await run_algorithm(
                algorithm=algorithm_map[centrality_type],
                graph_id=graph_id,
                **parameters,
            )

        @self.mcp.tool()
        async def community_detection(
            graph_id: str, method: str = "connected_components"
        ) -> Dict[str, Any]:
            """Detect communities in a graph.

            Args:
                graph_id: ID of the graph
                method: Community detection method ('connected_components', 'strongly_connected_components', 'weakly_connected_components')

            Returns:
                Dict with community information
            """
            valid_methods = [
                "connected_components",
                "strongly_connected_components",
                "weakly_connected_components",
            ]

            if method not in valid_methods:
                return {"error": f"Unsupported community detection method: {method}"}

            return await run_algorithm(algorithm=method, graph_id=graph_id)

        @self.mcp.tool()
        async def structural_analysis(
            graph_id: str, metrics: list[str] = None
        ) -> Dict[str, Any]:
            """Perform structural analysis of a graph.

            Args:
                graph_id: ID of the graph
                metrics: List of structural metrics to compute (optional)

            Returns:
                Dict with structural metrics
            """
            if metrics is None:
                metrics = ["density", "clustering", "transitivity"]

            available_metrics = {
                "density": "density",
                "clustering": "clustering",
                "transitivity": "transitivity",
                "triangle_count": "triangle_count",
                "diameter": "diameter",
                "radius": "radius",
                "center": "center",
                "periphery": "periphery",
            }

            results = {}
            errors = []

            for metric in metrics:
                if metric not in available_metrics:
                    errors.append(f"Unknown metric: {metric}")
                    continue

                try:
                    result = await run_algorithm(
                        algorithm=available_metrics[metric], graph_id=graph_id
                    )

                    if "error" in result:
                        errors.append(f"{metric}: {result['error']}")
                    else:
                        results[metric] = result.get("result", result)
                except Exception as e:
                    errors.append(f"{metric}: {str(e)}")

            response = {"graph_id": graph_id, "metrics": results}

            if errors:
                response["errors"] = errors

            return response

        @self.mcp.tool()
        async def flow_analysis(
            graph_id: str,
            source: str,
            target: str,
            capacity: str = "capacity",
            analysis_type: str = "maximum_flow",
        ) -> Dict[str, Any]:
            """Perform flow analysis on a graph.

            Args:
                graph_id: ID of the graph
                source: Source node
                target: Target node
                capacity: Edge attribute representing capacity
                analysis_type: Type of analysis ('maximum_flow', 'minimum_cut')

            Returns:
                Dict with flow analysis results
            """
            valid_types = ["maximum_flow", "minimum_cut"]

            if analysis_type not in valid_types:
                return {"error": f"Unsupported flow analysis type: {analysis_type}"}

            return await run_algorithm(
                algorithm=analysis_type,
                graph_id=graph_id,
                source=source,
                target=target,
                capacity=capacity,
            )

        @self.mcp.tool()
        async def spanning_tree_analysis(
            graph_id: str,
            weight: str = "weight",
            tree_type: str = "minimum",
            algorithm: str = "kruskal",
        ) -> Dict[str, Any]:
            """Compute spanning tree of a graph.

            Args:
                graph_id: ID of the graph
                weight: Edge attribute to use as weight
                tree_type: Type of spanning tree ('minimum', 'maximum')
                algorithm: Algorithm to use ('kruskal', 'prim')

            Returns:
                Dict with spanning tree information
            """
            algorithm_map = {
                "minimum": "minimum_spanning_tree",
                "maximum": "maximum_spanning_tree",
            }

            if tree_type not in algorithm_map:
                return {"error": f"Unsupported spanning tree type: {tree_type}"}

            return await run_algorithm(
                algorithm=algorithm_map[tree_type],
                graph_id=graph_id,
                weight=weight,
                algorithm=algorithm,
            )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the handler."""
        try:
            algorithm_service_health = await self._algorithm_service.health_check()
            validator_health = await self._validator.health_check()

            return {
                "healthy": (
                    algorithm_service_health.get("healthy", False)
                    and validator_health.get("healthy", False)
                ),
                "algorithm_service": algorithm_service_health,
                "validator": validator_health,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
