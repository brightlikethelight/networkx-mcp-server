#!/usr/bin/env python3
"""Minimal working NetworkX MCP Server - Critical Fix Version."""

import logging
from typing import Any

import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use compatibility layer instead of direct FastMCP import
from .compat.fastmcp_compat import FastMCPCompat

# Create the MCP server instance
mcp = FastMCPCompat(
    name="networkx-mcp", description="NetworkX graph operations via MCP"
)

# In-memory graph storage (simple for now)
graphs: dict[str, nx.Graph] = {}

# Import handlers for backward compatibility
try:
    from .handlers.graph_ops import graph_ops_handler
    from .handlers.algorithms import algorithms_handler
except ImportError:
    # Fallback if handlers are not available
    graph_ops_handler = None
    algorithms_handler = None




@mcp.tool(description="Create a new graph")
def create_graph(
    name: str, graph_type: str = "undirected", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a new NetworkX graph."""
    try:
        if name in graphs:
            return {"error": f"Graph '{name}' already exists"}

        # Create appropriate graph type
        if graph_type == "directed":
            G = nx.DiGraph()
        elif graph_type == "multi":
            G = nx.MultiGraph()
        elif graph_type == "multi_directed":
            G = nx.MultiDiGraph()
        else:
            G = nx.Graph()

        # Add any initial data
        if data:
            if "nodes" in data:
                G.add_nodes_from(data["nodes"])
            if "edges" in data:
                G.add_edges_from(data["edges"])

        graphs[name] = G

        return {
            "success": True,
            "name": name,
            "type": graph_type,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        return {"error": str(e)}


@mcp.tool(description="Add nodes to a graph")
def add_nodes(graph_name: str, nodes: list[Any]) -> dict[str, Any]:
    """Add nodes to an existing graph."""
    try:
        if graph_name not in graphs:
            return {"error": f"Graph '{graph_name}' not found"}

        G = graphs[graph_name]
        G.add_nodes_from(nodes)

        return {
            "success": True,
            "graph": graph_name,
            "nodes_added": len(nodes),
            "total_nodes": G.number_of_nodes(),
        }
    except Exception as e:
        logger.error(f"Error adding nodes: {e}")
        return {"error": str(e)}


@mcp.tool(description="Add edges to a graph")
def add_edges(graph_name: str, edges: list[list[Any]]) -> dict[str, Any]:
    """Add edges to an existing graph."""
    try:
        if graph_name not in graphs:
            return {"error": f"Graph '{graph_name}' not found"}

        G = graphs[graph_name]
        G.add_edges_from(edges)

        return {
            "success": True,
            "graph": graph_name,
            "edges_added": len(edges),
            "total_edges": G.number_of_edges(),
        }
    except Exception as e:
        logger.error(f"Error adding edges: {e}")
        return {"error": str(e)}


@mcp.tool(description="Get basic graph information")
def graph_info(graph_name: str) -> dict[str, Any]:
    """Get information about a graph."""
    try:
        if graph_name not in graphs:
            return {"error": f"Graph '{graph_name}' not found"}

        G = graphs[graph_name]

        info = {
            "name": graph_name,
            "type": G.__class__.__name__,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "is_directed": G.is_directed(),
            "is_multigraph": G.is_multigraph(),
            "density": nx.density(G) if G.number_of_nodes() > 0 else 0,
        }

        # Add more info for non-empty graphs
        if G.number_of_nodes() > 0:
            if not G.is_directed():
                info["is_connected"] = nx.is_connected(G)
                if info["is_connected"]:
                    info["diameter"] = nx.diameter(G)

        return info
    except Exception as e:
        logger.error(f"Error getting graph info: {e}")
        return {"error": str(e)}


@mcp.tool(description="List all available graphs")
def list_graphs() -> dict[str, Any]:
    """List all graphs in memory."""
    try:
        graph_list = []
        for name, G in graphs.items():
            graph_list.append(
                {
                    "name": name,
                    "type": G.__class__.__name__,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                }
            )

        return {"count": len(graphs), "graphs": graph_list}
    except Exception as e:
        logger.error(f"Error listing graphs: {e}")
        return {"error": str(e)}


@mcp.tool(description="Delete a graph")
def delete_graph(graph_name: str) -> dict[str, Any]:
    """Delete a graph from memory."""
    try:
        if graph_name not in graphs:
            return {"error": f"Graph '{graph_name}' not found"}

        del graphs[graph_name]

        return {"success": True, "deleted": graph_name, "remaining_graphs": len(graphs)}
    except Exception as e:
        logger.error(f"Error deleting graph: {e}")
        return {"error": str(e)}


@mcp.tool(description="Compute shortest path between two nodes")
def shortest_path(
    graph_name: str, source: Any, target: Any, weight: str | None = None
) -> dict[str, Any]:
    """Find shortest path between two nodes."""
    try:
        if graph_name not in graphs:
            return {"error": f"Graph '{graph_name}' not found"}

        G = graphs[graph_name]

        if source not in G:
            return {"error": f"Source node '{source}' not in graph"}
        if target not in G:
            return {"error": f"Target node '{target}' not in graph"}

        try:
            if weight:
                path = nx.shortest_path(G, source, target, weight=weight)
                length = nx.shortest_path_length(G, source, target, weight=weight)
            else:
                path = nx.shortest_path(G, source, target)
                length = nx.shortest_path_length(G, source, target)

            return {
                "success": True,
                "path": path,
                "length": length,
                "weighted": weight is not None,
            }
        except nx.NetworkXNoPath:
            return {
                "success": False,
                "error": f"No path exists between '{source}' and '{target}'",
            }
    except Exception as e:
        logger.error(f"Error finding shortest path: {e}")
        return {"error": str(e)}


@mcp.tool(description="Get node degree information")
def node_degree(graph_name: str, node: Any | None = None) -> dict[str, Any]:
    """Get degree information for nodes."""
    try:
        if graph_name not in graphs:
            return {"error": f"Graph '{graph_name}' not found"}

        G = graphs[graph_name]

        if node is not None:
            if node not in G:
                return {"error": f"Node '{node}' not in graph"}

            if G.is_directed():
                return {
                    "node": node,
                    "in_degree": G.in_degree(node),
                    "out_degree": G.out_degree(node),
                    "total_degree": G.degree(node),
                }
            else:
                return {"node": node, "degree": G.degree(node)}
        else:
            # Return degree for all nodes
            if G.is_directed():
                degrees = {
                    "in_degrees": dict(G.in_degree()),
                    "out_degrees": dict(G.out_degree()),
                    "total_degrees": dict(G.degree()),
                }
            else:
                degrees = {"degrees": dict(G.degree())}

            return degrees
    except Exception as e:
        logger.error(f"Error getting node degree: {e}")
        return {"error": str(e)}


class NetworkXMCPServer:
    """Simple wrapper class for backward compatibility."""

    def __init__(self):
        self.mcp = mcp
        self.graphs = graphs

    def run(self):
        """Run the server."""
        return main()


def main():
    """Run the minimal MCP server."""
    logger.info("Starting NetworkX MCP Server (Minimal Version)")
    logger.info("This is a critical fix version with basic functionality")

    try:
        # Run the server
        mcp.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
