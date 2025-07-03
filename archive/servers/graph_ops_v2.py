"""Graph Operations Handler v2 for NetworkX MCP Server.

This module provides the modernized MCP handlers that integrate with the
new service-oriented architecture using dependency injection.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from ...core.container import Container
from ...services.graph_service import GraphService
from ...validators.graph_validator import GraphValidator

logger = logging.getLogger(__name__)


class GraphOpsHandler:
    """Modern handler for basic graph operations using service architecture."""

    def __init__(self, mcp: FastMCP, container: Container):
        """Initialize the handler with MCP server and DI container."""
        self.mcp = mcp
        self.container = container
        self._graph_service: GraphService | None = None
        self._validator: GraphValidator | None = None

    async def initialize(self) -> None:
        """Initialize the handler and resolve dependencies."""
        self._graph_service = await self.container.resolve(GraphService)
        self._validator = await self.container.resolve(GraphValidator)
        self._register_tools()
        logger.info("Graph operations handler initialized")

    def _register_tools(self):
        """Register all graph operation tools."""

        @self.mcp.tool()
        async def create_graph(
            graph_id: str,
            graph_type: str = "Graph",
            description: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Create a new graph.

            Args:
                graph_id: Unique identifier for the graph
                graph_type: Type of graph - 'Graph', 'DiGraph', 'MultiGraph', 'MultiDiGraph'
                description: Optional description of the graph
                metadata: Optional metadata dictionary

            Returns:
                Dict with creation status and graph info
            """
            try:
                result = await self._graph_service.create_graph(
                    graph_id=graph_id,
                    graph_type=graph_type,
                    description=description,
                    metadata=metadata,
                )
                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to create graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def delete_graph(graph_id: str) -> dict[str, Any]:
            """Delete a graph from storage.

            Args:
                graph_id: ID of the graph to delete

            Returns:
                Dict with deletion status
            """
            try:
                result = await self._graph_service.delete_graph(graph_id)
                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to delete graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def list_graphs() -> dict[str, Any]:
            """List all available graphs.

            Returns:
                Dict with list of graphs and their basic info
            """
            try:
                graphs = await self._graph_service.list_graphs()
                return {"graphs": graphs, "count": len(graphs)}
            except Exception as e:
                logger.error(f"Failed to list graphs: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_graph_info(graph_id: str) -> dict[str, Any]:
            """Get detailed information about a graph.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with detailed graph information
            """
            try:
                info = await self._graph_service.get_graph_info(graph_id)
                return info
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to get graph info for {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def add_nodes(
            graph_id: str, nodes: list[dict[str, Any]]
        ) -> dict[str, Any]:
            """Add nodes to a graph.

            Args:
                graph_id: ID of the graph
                nodes: List of nodes to add. Each can be a string/int (node ID) or
                       dict with 'id' and optional 'attributes'

            Returns:
                Dict with operation status
            """
            try:
                # Validate request
                validation = await self._validator.validate_node_operation(
                    {"type": "add", "nodes": nodes}
                )

                if not validation.valid:
                    return {
                        "error": f"Validation failed: {', '.join(validation.errors)}"
                    }

                result = await self._graph_service.add_nodes(graph_id, nodes)
                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to add nodes to graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def remove_nodes(graph_id: str, nodes: list[str]) -> dict[str, Any]:
            """Remove nodes from a graph.

            Args:
                graph_id: ID of the graph
                nodes: List of node IDs to remove

            Returns:
                Dict with operation status
            """
            try:
                # Validate request
                validation = await self._validator.validate_node_operation(
                    {"type": "remove", "nodes": nodes}
                )

                if not validation.valid:
                    return {
                        "error": f"Validation failed: {', '.join(validation.errors)}"
                    }

                result = await self._graph_service.remove_nodes(graph_id, nodes)
                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to remove nodes from graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def add_edges(
            graph_id: str, edges: list[dict[str, Any]]
        ) -> dict[str, Any]:
            """Add edges to a graph.

            Args:
                graph_id: ID of the graph
                edges: List of edges to add. Each can be a tuple (source, target) or
                       dict with 'source', 'target' and optional 'attributes'

            Returns:
                Dict with operation status
            """
            try:
                # Validate request
                validation = await self._validator.validate_edge_operation(
                    {"type": "add", "edges": edges}
                )

                if not validation.valid:
                    return {
                        "error": f"Validation failed: {', '.join(validation.errors)}"
                    }

                result = await self._graph_service.add_edges(graph_id, edges)
                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to add edges to graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_nodes(graph_id: str) -> dict[str, Any]:
            """Get all nodes in a graph.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with nodes list
            """
            try:
                graph = await self._graph_service.get_graph(graph_id)
                nodes = [
                    {"id": node, "attributes": attrs}
                    for node, attrs in graph.nodes(data=True)
                ]
                return {"graph_id": graph_id, "nodes": nodes, "count": len(nodes)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to get nodes for graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_edges(graph_id: str) -> dict[str, Any]:
            """Get all edges in a graph.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with edges list
            """
            try:
                graph = await self._graph_service.get_graph(graph_id)
                edges = [
                    {"source": u, "target": v, "attributes": attrs}
                    for u, v, attrs in graph.edges(data=True)
                ]
                return {"graph_id": graph_id, "edges": edges, "count": len(edges)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to get edges for graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_neighbors(graph_id: str, node: str) -> dict[str, Any]:
            """Get neighbors of a specific node.

            Args:
                graph_id: ID of the graph
                node: Node ID to get neighbors for

            Returns:
                Dict with neighbors list
            """
            try:
                graph = await self._graph_service.get_graph(graph_id)

                if not graph.has_node(node):
                    return {"error": f"Node '{node}' not found in graph"}

                neighbors = list(graph.neighbors(node))
                return {
                    "graph_id": graph_id,
                    "node": node,
                    "neighbors": neighbors,
                    "count": len(neighbors),
                }
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(
                    f"Failed to get neighbors for node {node} in graph {graph_id}: {e}"
                )
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def get_node_degree(graph_id: str, node: str) -> dict[str, Any]:
            """Get degree of a specific node.

            Args:
                graph_id: ID of the graph
                node: Node ID to get degree for

            Returns:
                Dict with degree information
            """
            try:
                graph = await self._graph_service.get_graph(graph_id)

                if not graph.has_node(node):
                    return {"error": f"Node '{node}' not found in graph"}

                degree = graph.degree(node)
                result = {"graph_id": graph_id, "node": node, "degree": degree}

                # Add in/out degree for directed graphs
                if graph.is_directed():
                    result["in_degree"] = graph.in_degree(node)
                    result["out_degree"] = graph.out_degree(node)

                return result
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(
                    f"Failed to get degree for node {node} in graph {graph_id}: {e}"
                )
                return {"error": f"Internal error: {str(e)}"}

        @self.mcp.tool()
        async def validate_graph(graph_id: str) -> dict[str, Any]:
            """Validate graph data integrity.

            Args:
                graph_id: ID of the graph to validate

            Returns:
                Dict with validation results
            """
            try:
                graph = await self._graph_service.get_graph(graph_id)
                validation = await self._validator.validate_graph_data(graph)

                return {
                    "graph_id": graph_id,
                    "valid": validation.valid,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "metadata": validation.metadata,
                }
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to validate graph {graph_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

    async def health_check(self) -> dict[str, Any]:
        """Perform health check for the handler."""
        try:
            graph_service_health = await self._graph_service.health_check()
            validator_health = await self._validator.health_check()

            return {
                "healthy": (
                    graph_service_health.get("healthy", False)
                    and validator_health.get("healthy", False)
                ),
                "graph_service": graph_service_health,
                "validator": validator_health,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
