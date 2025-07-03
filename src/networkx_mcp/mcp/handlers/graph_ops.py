"""Graph Operations Handler for NetworkX MCP Server.

This module handles basic graph operations including creation, deletion,
modification, and querying of graphs.
"""

from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from ...compat.fastmcp_compat import FastMCPCompat as FastMCP


class GraphOpsHandler:
    """Handler for basic graph operations."""

    def __init__(self, mcp: FastMCP, graph_manager):
        """Initialize the handler with MCP server and graph manager."""
        self.mcp = mcp
        self.graph_manager = graph_manager
        self._register_tools()

    def _register_tools(self):
        """Register all graph operation tools."""

        @self.mcp.tool()
        async def create_graph(
            graph_id: str,
            graph_type: str = "undirected",
            from_data: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Create a new graph.

            Args:
                graph_id: Unique identifier for the graph
                graph_type: Type of graph - 'undirected', 'directed', 'multi_undirected', 'multi_directed'
                from_data: Optional data to initialize the graph (node-link format)

            Returns:
                Dict with creation status and graph info
            """
            if graph_id in self.graph_manager.graphs:
                return {"error": f"Graph '{graph_id}' already exists"}

            try:
                # Create appropriate graph type
                graph_types = {
                    "undirected": nx.Graph,
                    "directed": nx.DiGraph,
                    "multi_undirected": nx.MultiGraph,
                    "multi_directed": nx.MultiDiGraph,
                }

                if graph_type not in graph_types:
                    return {"error": f"Invalid graph type: {graph_type}"}

                GraphClass = graph_types[graph_type]

                if from_data:
                    # Create from provided data
                    G = json_graph.node_link_graph(
                        from_data,
                        directed=(graph_type in ["directed", "multi_directed"]),
                    )
                    if not isinstance(G, GraphClass):
                        # Convert to requested type
                        G = GraphClass(G)
                else:
                    # Create empty graph
                    G = GraphClass()

                self.graph_manager.add_graph(graph_id, G)

                return {
                    "status": "created",
                    "graph_id": graph_id,
                    "type": graph_type,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                }

            except Exception as e:
                return {"error": f"Failed to create graph: {str(e)}"}

        @self.mcp.tool()
        async def delete_graph(graph_id: str) -> dict[str, Any]:
            """Delete a graph from memory.

            Args:
                graph_id: ID of the graph to delete

            Returns:
                Dict with deletion status
            """
            if graph_id not in self.graph_manager.graphs:
                return {"error": f"Graph '{graph_id}' not found"}

            self.graph_manager.delete_graph(graph_id)
            return {"status": "deleted", "graph_id": graph_id}

        @self.mcp.tool()
        async def list_graphs() -> dict[str, Any]:
            """List all available graphs.

            Returns:
                Dict with list of graph IDs and their basic info
            """
            graphs_info = []
            for graph_id in self.graph_manager.list_graphs():
                G = self.graph_manager.get_graph(graph_id)
                if G:
                    graphs_info.append(
                        {
                            "id": graph_id,
                            "type": G.__class__.__name__,
                            "nodes": G.number_of_nodes(),
                            "edges": G.number_of_edges(),
                        }
                    )

            return {"graphs": graphs_info, "count": len(graphs_info)}

        @self.mcp.tool()
        async def get_graph_info(graph_id: str) -> dict[str, Any]:
            """Get detailed information about a graph.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with detailed graph information
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            info = {
                "graph_id": graph_id,
                "type": G.__class__.__name__,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "is_directed": G.is_directed(),
                "is_multigraph": G.is_multigraph(),
                "density": nx.density(G) if G.number_of_nodes() > 0 else 0,
            }

            # Add connectivity info
            if G.number_of_nodes() > 0:
                if G.is_directed():
                    info["is_weakly_connected"] = nx.is_weakly_connected(G)
                    info["is_strongly_connected"] = nx.is_strongly_connected(G)
                    info["number_weakly_connected_components"] = (
                        nx.number_weakly_connected_components(G)
                    )
                    info["number_strongly_connected_components"] = (
                        nx.number_strongly_connected_components(G)
                    )
                else:
                    info["is_connected"] = nx.is_connected(G)
                    info["number_connected_components"] = (
                        nx.number_connected_components(G)
                    )

            # Sample nodes and edges
            info["sample_nodes"] = list(G.nodes())[:10]
            info["sample_edges"] = list(G.edges())[:10]

            return info

        @self.mcp.tool()
        async def add_nodes(
            graph_id: str,
            nodes: list[str | int],
            attributes: dict[str, dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            """Add nodes to a graph.

            Args:
                graph_id: ID of the graph
                nodes: List of node identifiers
                attributes: Optional dict mapping node IDs to their attributes

            Returns:
                Dict with operation status
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if attributes:
                    for node in nodes:
                        if node in attributes:
                            G.add_node(node, **attributes[node])
                        else:
                            G.add_node(node)
                else:
                    G.add_nodes_from(nodes)

                return {
                    "status": "nodes_added",
                    "count": len(nodes),
                    "total_nodes": G.number_of_nodes(),
                }

            except Exception as e:
                return {"error": f"Failed to add nodes: {str(e)}"}

        @self.mcp.tool()
        async def add_edges(
            graph_id: str,
            edges: list[list[Any] | tuple[Any, Any]],
            attributes: dict[str, dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            """Add edges to a graph.

            Args:
                graph_id: ID of the graph
                edges: List of edges as [source, target] or (source, target)
                attributes: Optional dict mapping edge tuples to their attributes

            Returns:
                Dict with operation status
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if attributes:
                    for edge in edges:
                        u, v = edge[0], edge[1]
                        edge_key = (u, v)
                        if edge_key in attributes:
                            G.add_edge(u, v, **attributes[edge_key])
                        else:
                            G.add_edge(u, v)
                else:
                    G.add_edges_from(edges)

                return {
                    "status": "edges_added",
                    "count": len(edges),
                    "total_edges": G.number_of_edges(),
                }

            except Exception as e:
                return {"error": f"Failed to add edges: {str(e)}"}

        @self.mcp.tool()
        async def remove_nodes(graph_id: str, nodes: list[str | int]) -> dict[str, Any]:
            """Remove nodes from a graph.

            Args:
                graph_id: ID of the graph
                nodes: List of node identifiers to remove

            Returns:
                Dict with operation status
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                # Filter out nodes that don't exist
                existing_nodes = [n for n in nodes if n in G]
                G.remove_nodes_from(existing_nodes)

                return {
                    "status": "nodes_removed",
                    "removed_count": len(existing_nodes),
                    "remaining_nodes": G.number_of_nodes(),
                }

            except Exception as e:
                return {"error": f"Failed to remove nodes: {str(e)}"}

        @self.mcp.tool()
        async def remove_edges(
            graph_id: str, edges: list[list[Any] | tuple[Any, Any]]
        ) -> dict[str, Any]:
            """Remove edges from a graph.

            Args:
                graph_id: ID of the graph
                edges: List of edges to remove as [source, target]

            Returns:
                Dict with operation status
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                # Filter out edges that don't exist
                existing_edges = [(u, v) for u, v in edges if G.has_edge(u, v)]
                G.remove_edges_from(existing_edges)

                return {
                    "status": "edges_removed",
                    "removed_count": len(existing_edges),
                    "remaining_edges": G.number_of_edges(),
                }

            except Exception as e:
                return {"error": f"Failed to remove edges: {str(e)}"}

        @self.mcp.tool()
        async def clear_graph(graph_id: str) -> dict[str, Any]:
            """Clear all nodes and edges from a graph.

            Args:
                graph_id: ID of the graph to clear

            Returns:
                Dict with operation status
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                G.clear()
                return {
                    "status": "cleared",
                    "graph_id": graph_id,
                    "nodes": 0,
                    "edges": 0,
                }

            except Exception as e:
                return {"error": f"Failed to clear graph: {str(e)}"}

        @self.mcp.tool()
        async def subgraph_extraction(
            graph_id: str,
            nodes: list[str | int] | None = None,
            k_hop: int | None = None,
            center_node: str | int | None = None,
            new_graph_id: str | None = None,
        ) -> dict[str, Any]:
            """Extract a subgraph from an existing graph.

            Args:
                graph_id: ID of the source graph
                nodes: List of nodes to include in subgraph
                k_hop: Extract k-hop neighborhood (requires center_node)
                center_node: Center node for k-hop extraction
                new_graph_id: Optional ID for the new subgraph

            Returns:
                Dict with subgraph information
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if k_hop is not None and center_node is not None:
                    # Extract k-hop neighborhood
                    if center_node not in G:
                        return {"error": f"Center node '{center_node}' not found"}

                    nodes_set = {center_node}
                    for _ in range(k_hop):
                        neighbors = set()
                        for node in nodes_set:
                            neighbors.update(G.neighbors(node))
                        nodes_set.update(neighbors)

                    subgraph = G.subgraph(nodes_set).copy()

                elif nodes:
                    # Extract subgraph with specified nodes
                    valid_nodes = [n for n in nodes if n in G]
                    if not valid_nodes:
                        return {"error": "None of the specified nodes exist in graph"}

                    subgraph = G.subgraph(valid_nodes).copy()

                else:
                    return {
                        "error": "Must specify either nodes or k_hop with center_node"
                    }

                # Save subgraph if ID provided
                if new_graph_id:
                    if new_graph_id in self.graph_manager.graphs:
                        return {"error": f"Graph '{new_graph_id}' already exists"}
                    self.graph_manager.add_graph(new_graph_id, subgraph)

                result = {
                    "nodes": subgraph.number_of_nodes(),
                    "edges": subgraph.number_of_edges(),
                    "node_list": list(subgraph.nodes()),
                }

                if new_graph_id:
                    result["new_graph_id"] = new_graph_id

                return result

            except Exception as e:
                return {"error": f"Failed to extract subgraph: {str(e)}"}


__all__ = ["GraphOpsHandler"]
