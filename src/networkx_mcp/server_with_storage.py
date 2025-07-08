#!/usr/bin/env python3
"""NetworkX MCP Server with Configurable Storage Backend."""

import asyncio
import logging
import os
import signal
from typing import Any

import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import sophisticated components
from .core.graph_operations import GraphManager
from .core.algorithms import GraphAlgorithms
from .core.storage_manager import StorageManager

# Use compatibility layer
from .compat.fastmcp_compat import FastMCPCompat

# Import security components
from .security.input_validation import (
    validate_id,
    validate_node_list,
    validate_edge_list,
    validate_graph_type,
    safe_error_message,
    ValidationError,
)
from .security.resource_limits import (
    with_resource_limits,
    check_graph_size,
    check_operation_feasibility,
    estimate_graph_size_mb,
    get_resource_status,
    ResourceLimitError,
)

# Initialize core components
graph_manager = GraphManager()
graph_algorithms = GraphAlgorithms()
storage_manager = StorageManager(graph_manager)

# Global flag for storage initialization
storage_initialized = False

# Create the MCP server instance
mcp = FastMCPCompat(
    name="networkx-mcp", 
    description="NetworkX graph operations via MCP - With Storage Support"
)

# For backward compatibility - expose graphs dict
class GraphsProxy:
    """Proxy to make graph_manager.graphs accessible as module.graphs."""
    def __getitem__(self, key):
        return graph_manager.graphs[key]
    
    def __setitem__(self, key, value):
        graph_manager.graphs[key] = value
        # Trigger async save
        if storage_initialized:
            asyncio.create_task(storage_manager.save_graph(key))
        
    def __contains__(self, key):
        return key in graph_manager.graphs
        
    def __iter__(self):
        return iter(graph_manager.graphs)
        
    def items(self):
        return graph_manager.graphs.items()
        
    def keys(self):
        return graph_manager.graphs.keys()
        
    def values(self):
        return graph_manager.graphs.values()
        
    def get(self, key, default=None):
        return graph_manager.graphs.get(key, default)

graphs = GraphsProxy()


async def initialize_storage():
    """Initialize storage backend on startup."""
    global storage_initialized
    
    try:
        logger.info("Initializing storage backend...")
        await storage_manager.initialize()
        
        # Log storage health
        health = await storage_manager.get_storage_stats()
        logger.info(f"Storage backend initialized: {health.get('backend', 'unknown')}")
        logger.info(f"Storage health: {health}")
        
        storage_initialized = True
        
        # Log loaded graphs
        if graph_manager.graphs:
            logger.info(f"Loaded {len(graph_manager.graphs)} graphs from storage")
            for graph_id, graph in graph_manager.graphs.items():
                logger.info(f"  - {graph_id}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        else:
            logger.info("No existing graphs found in storage")
            
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        logger.warning("Continuing without storage - data will not persist!")
        storage_initialized = False


async def shutdown_storage():
    """Shutdown storage backend gracefully."""
    if storage_initialized:
        try:
            logger.info("Shutting down storage backend...")
            await storage_manager.close()
            logger.info("Storage backend shutdown complete")
        except Exception as e:
            logger.error(f"Error during storage shutdown: {e}")


@mcp.tool(description="Get storage backend status and statistics")
async def storage_status() -> dict[str, Any]:
    """Get current storage backend status and statistics."""
    try:
        if not storage_initialized:
            return {
                "status": "not_initialized",
                "message": "Storage backend not initialized"
            }
            
        stats = await storage_manager.get_storage_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting storage status: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Create a new graph")
@with_resource_limits
async def create_graph(
    name: str, graph_type: str = "undirected", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a new NetworkX graph using GraphManager."""
    try:
        # Validate inputs
        safe_name = validate_id(name, "Graph name")
        safe_type = validate_graph_type(graph_type)
        
        # Map user-friendly types to NetworkX types
        type_map = {
            "undirected": "Graph",
            "directed": "DiGraph",
            "multi": "MultiGraph",
            "multi_directed": "MultiDiGraph",
        }
        
        nx_type = type_map.get(safe_type, "Graph")
        
        # Create graph using GraphManager
        result = graph_manager.create_graph(
            graph_id=safe_name,
            graph_type=nx_type
        )
        
        # Add initial data if provided
        if data and result.get("created"):
            graph = graph_manager.get_graph(safe_name)
            
            if "nodes" in data:
                validated_nodes = validate_node_list(data["nodes"])
                graph_manager.add_nodes_from(safe_name, validated_nodes)
                
            if "edges" in data:
                validated_edges = validate_edge_list(data["edges"]) 
                graph_manager.add_edges_from(safe_name, validated_edges)
            
            # Check graph size after adding data
            check_graph_size(graph)
        
        # Save to storage if initialized
        if storage_initialized:
            await storage_manager.save_graph(safe_name)
        
        # Get final graph info
        graph_info = graph_manager.get_graph_info(safe_name)
        
        return {
            "success": True,
            "name": safe_name,
            "type": safe_type,
            "nodes": graph_info["num_nodes"],
            "edges": graph_info["num_edges"],
            "estimated_size_mb": estimate_graph_size_mb(graph_manager.get_graph(safe_name)),
            "persisted": storage_initialized,
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error in create_graph: {e}")
        return {"error": str(e)}
    except ResourceLimitError as e:
        logger.warning(f"Resource limit in create_graph: {e}")
        return {"error": str(e)}
    except ValueError as e:
        # GraphManager raises ValueError for duplicate graphs
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Delete a graph")
@with_resource_limits
async def delete_graph(graph_name: str) -> dict[str, Any]:
    """Delete a graph using GraphManager."""
    try:
        # Validate input
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        # Delete using GraphManager
        result = graph_manager.delete_graph(safe_graph_name)
        
        # Delete from storage if initialized
        if storage_initialized:
            await storage_manager.delete_graph(safe_graph_name)
        
        return {
            "success": True,
            "deleted": safe_graph_name,
            "remaining_graphs": len(graph_manager.graphs)
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error in delete_graph: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error deleting graph: {e}")
        return {"error": safe_error_message(e)}


# Copy all the other tool functions from the original server.py
# (add_nodes, add_edges, graph_info, list_graphs, shortest_path, etc.)
# But add storage save calls after modifications

@mcp.tool(description="Add nodes to a graph")
@with_resource_limits
async def add_nodes(graph_name: str, nodes: list[Any]) -> dict[str, Any]:
    """Add nodes to an existing graph using GraphManager."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        validated_nodes = validate_node_list(nodes)
        
        # Add nodes using GraphManager
        result = graph_manager.add_nodes_from(safe_graph_name, validated_nodes)
        
        # Check graph size after adding nodes
        graph = graph_manager.get_graph(safe_graph_name)
        check_graph_size(graph)
        
        # Save to storage if initialized
        if storage_initialized:
            await storage_manager.save_graph(safe_graph_name)
        
        return {
            "success": True,
            "graph": safe_graph_name,
            "nodes_added": result["nodes_added"],
            "total_nodes": result["total_nodes"],
            "estimated_size_mb": estimate_graph_size_mb(graph),
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error in add_nodes: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error adding nodes: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Add edges to a graph")
@with_resource_limits
async def add_edges(graph_name: str, edges: list[list[Any]]) -> dict[str, Any]:
    """Add edges to an existing graph using GraphManager."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        validated_edges = validate_edge_list(edges)
        
        # Add edges using GraphManager
        result = graph_manager.add_edges_from(safe_graph_name, validated_edges)
        
        # Check graph size after adding edges
        graph = graph_manager.get_graph(safe_graph_name)
        check_graph_size(graph)
        
        # Save to storage if initialized
        if storage_initialized:
            await storage_manager.save_graph(safe_graph_name)
        
        return {
            "success": True,
            "graph": safe_graph_name,
            "edges_added": result["edges_added"], 
            "total_edges": result["total_edges"],
            "estimated_size_mb": estimate_graph_size_mb(graph),
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error in add_edges: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error adding edges: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Get basic graph information")
@with_resource_limits
def graph_info(graph_name: str) -> dict[str, Any]:
    """Get information about a graph using GraphManager."""
    try:
        # Validate input
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        # Get info from GraphManager
        info = graph_manager.get_graph_info(safe_graph_name)
        
        # Convert to expected format
        return {
            "name": safe_graph_name,
            "type": info["graph_type"],
            "nodes": info["num_nodes"],
            "edges": info["num_edges"],
            "is_directed": info["is_directed"],
            "is_multigraph": info["is_multigraph"],
            "density": info["density"],
            "degree_stats": info.get("degree_stats", {}),
            "metadata": info.get("metadata", {}),
            "persisted": storage_initialized,
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error in graph_info: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error getting graph info: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="List all available graphs")
@with_resource_limits
def list_graphs() -> dict[str, Any]:
    """List all graphs using GraphManager."""
    try:
        graph_list = graph_manager.list_graphs()
        
        # Convert to expected format
        return {
            "count": len(graph_list),
            "graphs": [
                {
                    "name": g["graph_id"],
                    "type": g["graph_type"],
                    "nodes": g["num_nodes"],
                    "edges": g["num_edges"],
                    "created_at": g.get("metadata", {}).get("created_at"),
                }
                for g in graph_list
            ],
            "storage_backend": storage_manager.storage_backend.__class__.__name__ if storage_initialized else "None",
        }
    except Exception as e:
        logger.error(f"Error listing graphs: {e}")
        return {"error": safe_error_message(e)}


# Include all the algorithm tools from the original server
# (shortest_path, node_degree, connected_components, etc.)
# These don't need storage calls as they don't modify graphs

@mcp.tool(description="Compute shortest path between two nodes")
@with_resource_limits
def shortest_path(
    graph_name: str, source: Any, target: Any, weight: str | None = None
) -> dict[str, Any]:
    """Find shortest path using GraphAlgorithms."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        # Get graph
        graph = graph_manager.get_graph(safe_graph_name)
        
        # Check operation feasibility
        check_operation_feasibility(graph, "shortest_path")
        
        # Allow integers for node IDs
        if isinstance(source, int):
            safe_source = source
        else:
            safe_source = validate_id(source, "Source node")
            
        if isinstance(target, int):
            safe_target = target
        else:
            safe_target = validate_id(target, "Target node")
        
        # Use GraphAlgorithms
        result = graph_algorithms.shortest_path(
            graph=graph,
            source=safe_source,
            target=safe_target,
            weight=weight,
            method="dijkstra"
        )
        
        return {
            "success": True,
            "path": result["path"],
            "length": result["length"],
            "weighted": weight is not None,
        }
        
    except ValidationError as e:
        logger.warning(f"Validation error in shortest_path: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except ValueError as e:
        return {"error": str(e)}
    except nx.NetworkXNoPath:
        return {
            "success": False,
            "error": f"No path exists between '{source}' and '{target}'",
        }
    except Exception as e:
        logger.error(f"Error finding shortest path: {e}")
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
    """Wrapper class for backward compatibility."""
    
    def __init__(self):
        self.mcp = mcp
        self.graphs = graphs
        self.graph_manager = graph_manager
        self.graph_algorithms = graph_algorithms
        self.storage_manager = storage_manager
    
    async def run(self):
        """Run the server with async support."""
        # Initialize storage
        await initialize_storage()
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        
        def handle_shutdown(sig):
            logger.info(f"Received signal {sig}, shutting down...")
            asyncio.create_task(shutdown_and_exit())
        
        async def shutdown_and_exit():
            await shutdown_storage()
            loop.stop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
        
        # Run the MCP server
        try:
            await mcp.run_async()
        finally:
            await shutdown_storage()


async def main():
    """Run the integrated MCP server with storage."""
    logger.info("Starting NetworkX MCP Server with Storage Support")
    logger.info("Using GraphManager, GraphAlgorithms, and StorageManager")
    
    server = NetworkXMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())