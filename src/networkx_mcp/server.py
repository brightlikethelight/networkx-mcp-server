#!/usr/bin/env python3
"""Minimal working NetworkX MCP Server - Security Hardened Version."""

import logging
from typing import Any

import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use compatibility layer instead of direct FastMCP import
from .compat.fastmcp_compat import FastMCPCompat

# Import security validation
from .security.input_validation import (
    validate_id,
    validate_node_list,
    validate_edge_list,
    validate_attributes,
    validate_graph_type,
    safe_error_message,
    ValidationError,
    MAX_NODES_PER_REQUEST,
    MAX_EDGES_PER_REQUEST,
)

# Import resource limits
from .security.resource_limits import (
    with_resource_limits,
    check_memory_limit,
    check_graph_size,
    check_operation_feasibility,
    estimate_graph_size_mb,
    get_resource_status,
    ResourceLimitError,
    LIMITS,
)

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
@with_resource_limits
def create_graph(
    name: str, graph_type: str = "undirected", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a new NetworkX graph with security validation and resource limits."""
    try:
        # Validate inputs
        safe_name = validate_id(name, "Graph name")
        safe_type = validate_graph_type(graph_type)
        
        if safe_name in graphs:
            return {"error": f"Graph '{safe_name}' already exists"}

        # Create appropriate graph type
        if safe_type == "directed":
            G = nx.DiGraph()
        elif safe_type == "multi":
            G = nx.MultiGraph()
        elif safe_type == "multi_directed":
            G = nx.MultiDiGraph()
        else:
            G = nx.Graph()

        # Add any initial data with validation
        if data:
            if "nodes" in data:
                validated_nodes = validate_node_list(data["nodes"])
                G.add_nodes_from(validated_nodes)
            if "edges" in data:
                validated_edges = validate_edge_list(data["edges"])
                G.add_edges_from(validated_edges)

        # Check graph size before storing
        check_graph_size(G)
        
        graphs[safe_name] = G

        return {
            "success": True,
            "name": safe_name,
            "type": safe_type,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "estimated_size_mb": estimate_graph_size_mb(G),
        }
    except ValidationError as e:
        logger.warning(f"Validation error in create_graph: {e}")
        return {"error": str(e)}
    except ResourceLimitError as e:
        logger.warning(f"Resource limit in create_graph: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Add nodes to a graph")
@with_resource_limits
def add_nodes(graph_name: str, nodes: list[Any]) -> dict[str, Any]:
    """Add nodes to an existing graph with security validation and resource limits."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        validated_nodes = validate_node_list(nodes, MAX_NODES_PER_REQUEST)
        
        if safe_graph_name not in graphs:
            return {"error": f"Graph '{safe_graph_name}' not found"}

        G = graphs[safe_graph_name]
        
        # Check if adding these nodes would exceed limits
        total_nodes = G.number_of_nodes() + len(validated_nodes)
        if total_nodes > MAX_NODES_PER_REQUEST * 10:  # Reasonable total limit
            return {"error": f"Graph would exceed maximum node limit"}

        G.add_nodes_from(validated_nodes)
        
        # Check graph size after adding nodes
        check_graph_size(G)

        return {
            "success": True,
            "graph": safe_graph_name,
            "nodes_added": len(validated_nodes),
            "total_nodes": G.number_of_nodes(),
            "estimated_size_mb": estimate_graph_size_mb(G),
        }
    except ValidationError as e:
        logger.warning(f"Validation error in add_nodes: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error adding nodes: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Add edges to a graph")
@with_resource_limits
def add_edges(graph_name: str, edges: list[list[Any]]) -> dict[str, Any]:
    """Add edges to an existing graph with security validation and resource limits."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        validated_edges = validate_edge_list(edges, MAX_EDGES_PER_REQUEST)
        
        if safe_graph_name not in graphs:
            return {"error": f"Graph '{safe_graph_name}' not found"}

        G = graphs[safe_graph_name]
        
        # Check if adding these edges would exceed limits
        total_edges = G.number_of_edges() + len(validated_edges)
        if total_edges > MAX_EDGES_PER_REQUEST * 10:  # Reasonable total limit
            return {"error": f"Graph would exceed maximum edge limit"}

        G.add_edges_from(validated_edges)
        
        # Check graph size after adding edges
        check_graph_size(G)

        return {
            "success": True,
            "graph": safe_graph_name,
            "edges_added": len(validated_edges),
            "total_edges": G.number_of_edges(),
            "estimated_size_mb": estimate_graph_size_mb(G),
        }
    except ValidationError as e:
        logger.warning(f"Validation error in add_edges: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error adding edges: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Get basic graph information")
@with_resource_limits
def graph_info(graph_name: str) -> dict[str, Any]:
    """Get information about a graph with security validation and resource limits."""
    try:
        # Validate input
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        if safe_graph_name not in graphs:
            return {"error": f"Graph '{safe_graph_name}' not found"}

        G = graphs[safe_graph_name]

        info = {
            "name": safe_graph_name,
            "type": G.__class__.__name__,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "is_directed": G.is_directed(),
            "is_multigraph": G.is_multigraph(),
            "density": nx.density(G) if G.number_of_nodes() > 0 else 0,
        }

        # Add more info for non-empty graphs (with safety checks for large graphs)
        if G.number_of_nodes() > 0 and G.number_of_nodes() < 10000:
            if not G.is_directed():
                try:
                    info["is_connected"] = nx.is_connected(G)
                    if info["is_connected"] and G.number_of_nodes() < 1000:
                        info["diameter"] = nx.diameter(G)
                except Exception:
                    # Skip expensive operations on large graphs
                    pass

        return info
    except ValidationError as e:
        logger.warning(f"Validation error in graph_info: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error getting graph info: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="List all available graphs")
@with_resource_limits
def list_graphs() -> dict[str, Any]:
    """List all graphs in memory with resource limits."""
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
@with_resource_limits
def delete_graph(graph_name: str) -> dict[str, Any]:
    """Delete a graph from memory with security validation and resource limits."""
    try:
        # Validate input
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        if safe_graph_name not in graphs:
            return {"error": f"Graph '{safe_graph_name}' not found"}

        del graphs[safe_graph_name]

        return {"success": True, "deleted": safe_graph_name, "remaining_graphs": len(graphs)}
    except ValidationError as e:
        logger.warning(f"Validation error in delete_graph: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error deleting graph: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Compute shortest path between two nodes")
@with_resource_limits
def shortest_path(
    graph_name: str, source: Any, target: Any, weight: str | None = None
) -> dict[str, Any]:
    """Find shortest path between two nodes with security validation and resource limits."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        # Allow integers for node IDs
        if isinstance(source, int):
            safe_source = source
        else:
            safe_source = validate_id(source, "Source node")
            
        if isinstance(target, int):
            safe_target = target
        else:
            safe_target = validate_id(target, "Target node")
        
        if weight is not None:
            safe_weight = validate_id(weight, "Weight attribute")
        else:
            safe_weight = None
        
        if safe_graph_name not in graphs:
            return {"error": f"Graph '{safe_graph_name}' not found"}

        G = graphs[safe_graph_name]
        
        # Check if operation is feasible for graph size
        check_operation_feasibility(G, "shortest_path")

        if safe_source not in G:
            return {"error": f"Source node '{safe_source}' not in graph"}
        if safe_target not in G:
            return {"error": f"Target node '{safe_target}' not in graph"}

        try:
            if safe_weight:
                path = nx.shortest_path(G, safe_source, safe_target, weight=safe_weight)
                length = nx.shortest_path_length(G, safe_source, safe_target, weight=safe_weight)
            else:
                path = nx.shortest_path(G, safe_source, safe_target)
                length = nx.shortest_path_length(G, safe_source, safe_target)

            return {
                "success": True,
                "path": path,
                "length": length,
                "weighted": safe_weight is not None,
            }
        except nx.NetworkXNoPath:
            return {
                "success": False,
                "error": f"No path exists between '{safe_source}' and '{safe_target}'",
            }
    except ValidationError as e:
        logger.warning(f"Validation error in shortest_path: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error finding shortest path: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Get node degree information")
@with_resource_limits
def node_degree(graph_name: str, node: Any | None = None) -> dict[str, Any]:
    """Get degree information for nodes with security validation and resource limits."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        if node is not None:
            # Allow integers for node IDs
            if isinstance(node, int):
                safe_node = node
            else:
                safe_node = validate_id(node, "Node")
        else:
            safe_node = None
        
        if safe_graph_name not in graphs:
            return {"error": f"Graph '{safe_graph_name}' not found"}

        G = graphs[safe_graph_name]

        if safe_node is not None:
            if safe_node not in G:
                return {"error": f"Node '{safe_node}' not in graph"}

            if G.is_directed():
                return {
                    "node": safe_node,
                    "in_degree": G.in_degree(safe_node),
                    "out_degree": G.out_degree(safe_node),
                    "total_degree": G.degree(safe_node),
                }
            else:
                return {"node": safe_node, "degree": G.degree(safe_node)}
        else:
            # Limit output for large graphs
            if G.number_of_nodes() > 10000:
                return {"error": "Graph too large to return all degrees. Please query specific nodes."}
                
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
    except ValidationError as e:
        logger.warning(f"Validation error in node_degree: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error getting node degree: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Get current resource usage and limits")
def resource_status() -> dict[str, Any]:
    """Get current resource usage and limits."""
    try:
        return get_resource_status()
    except Exception as e:
        logger.error(f"Error getting resource status: {e}")
        return {"error": safe_error_message(e)}


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
