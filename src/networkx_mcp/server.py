#!/usr/bin/env python3
"""NetworkX MCP Server - Integrated Version using GraphManager."""

import json
import logging
import os
import sys
import time
from typing import Any

import networkx as nx

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import structured logging first
from .logging import get_logger, configure_logging as structured_configure_logging, CorrelationContext, timed_operation, correlation_middleware

# Configure structured logging
structured_configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format_type=os.getenv("LOG_FORMAT", "colored"),
    output_file=os.getenv("LOG_FILE"),
    include_correlation=True,
    include_context=True
)

# Import configuration  
try:
    from config import get_settings
    settings = get_settings()
except ImportError:
    # Fallback if config module not available
    settings = None

logger = get_logger(__name__)

# Import sophisticated components
from .core.graph_operations import GraphManager
from .core.algorithms import GraphAlgorithms

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

# Create the MCP server instance
mcp = FastMCPCompat(
    name="networkx-mcp", 
    description="NetworkX graph operations via MCP - Integrated Version"
)

# For backward compatibility - expose graphs dict
@property
def graphs_property():
    """Property to access graphs for backward compatibility."""
    return graph_manager.graphs

# Create a module-level attribute that acts like the old graphs dict
class GraphsProxy:
    """Proxy to make graph_manager.graphs accessible as module.graphs."""
    def __getitem__(self, key):
        return graph_manager.graphs[key]
    
    def __setitem__(self, key, value):
        graph_manager.graphs[key] = value
        
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


@mcp.tool(description="Create a new graph")
@correlation_middleware
@timed_operation("graph.create", log_args=True)
@with_resource_limits
def create_graph(
    name: str, graph_type: str = "undirected", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a new NetworkX graph using GraphManager."""
    logger.info("Creating new graph", extra={
        "graph_name": name,
        "graph_type": graph_type,
        "has_initial_data": data is not None
    })
    
    try:
        # Validate inputs
        safe_name = validate_id(name, "Graph name")
        safe_type = validate_graph_type(graph_type)
        
        logger.debug("Graph inputs validated", extra={
            "safe_name": safe_name,
            "safe_type": safe_type
        })
        
        # Map user-friendly types to NetworkX types
        type_map = {
            "undirected": "Graph",
            "directed": "DiGraph",
            "multi": "MultiGraph",
            "multi_directed": "MultiDiGraph",
        }
        
        nx_type = type_map.get(safe_type, "Graph")
        
        # Create graph using GraphManager
        logger.debug("Creating graph with GraphManager", extra={
            "graph_id": safe_name,
            "nx_type": nx_type
        })
        
        result = graph_manager.create_graph(
            graph_id=safe_name,
            graph_type=nx_type
        )
        
        # Add initial data if provided
        if data and result.get("created"):
            logger.debug("Adding initial data to graph", extra={
                "graph_name": safe_name,
                "has_nodes": "nodes" in data,
                "has_edges": "edges" in data
            })
            
            graph = graph_manager.get_graph(safe_name)
            
            if "nodes" in data:
                validated_nodes = validate_node_list(data["nodes"])
                graph_manager.add_nodes_from(safe_name, validated_nodes)
                logger.debug("Added initial nodes", extra={
                    "graph_name": safe_name,
                    "node_count": len(validated_nodes)
                })
                
            if "edges" in data:
                validated_edges = validate_edge_list(data["edges"]) 
                graph_manager.add_edges_from(safe_name, validated_edges)
                logger.debug("Added initial edges", extra={
                    "graph_name": safe_name,
                    "edge_count": len(validated_edges)
                })
            
            # Check graph size after adding data
            check_graph_size(graph)
        
        # Get final graph info
        graph_info = graph_manager.get_graph_info(safe_name)
        
        result_data = {
            "success": True,
            "name": safe_name,
            "type": safe_type,
            "nodes": graph_info["num_nodes"],
            "edges": graph_info["num_edges"],
            "estimated_size_mb": estimate_graph_size_mb(graph_manager.get_graph(safe_name)),
        }
        
        logger.info("Graph created successfully", extra={
            "graph_name": safe_name,
            "graph_type": safe_type,
            "final_nodes": graph_info["num_nodes"],
            "final_edges": graph_info["num_edges"],
            "size_mb": result_data["estimated_size_mb"]
        })
        
        return result_data
        
    except ValidationError as e:
        logger.warning("Validation error in create_graph", extra={
            "error": str(e),
            "graph_name": name,
            "graph_type": graph_type
        })
        return {"error": str(e)}
    except ResourceLimitError as e:
        logger.warning("Resource limit exceeded in create_graph", extra={
            "error": str(e),
            "graph_name": name,
            "graph_type": graph_type
        })
        return {"error": str(e)}
    except ValueError as e:
        # GraphManager raises ValueError for duplicate graphs
        logger.warning("Graph creation failed", extra={
            "error": str(e),
            "graph_name": name,
            "graph_type": graph_type,
            "reason": "duplicate_or_invalid"
        })
        return {"error": str(e)}
    except Exception as e:
        logger.error("Unexpected error creating graph", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "graph_name": name,
            "graph_type": graph_type
        }, exc_info=True)
        return {"error": safe_error_message(e)}


@mcp.tool(description="Add nodes to a graph")
@correlation_middleware
@timed_operation("graph.add_nodes", log_args=True)
@with_resource_limits
def add_nodes(graph_name: str, nodes: list[Any]) -> dict[str, Any]:
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
def add_edges(graph_name: str, edges: list[list[Any]]) -> dict[str, Any]:
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
            ]
        }
    except Exception as e:
        logger.error(f"Error listing graphs: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Delete a graph")
@with_resource_limits
def delete_graph(graph_name: str) -> dict[str, Any]:
    """Delete a graph using GraphManager."""
    try:
        # Validate input
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        # Delete using GraphManager
        result = graph_manager.delete_graph(safe_graph_name)
        
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


@mcp.tool(description="Compute shortest path between two nodes")
@correlation_middleware
@timed_operation("graph.shortest_path", log_args=True)
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


@mcp.tool(description="Get node degree information")
@with_resource_limits
def node_degree(graph_name: str, node: Any | None = None) -> dict[str, Any]:
    """Get degree information for nodes."""
    try:
        # Validate inputs
        safe_graph_name = validate_id(graph_name, "Graph name")
        
        # Get graph
        graph = graph_manager.get_graph(safe_graph_name)
        
        if node is not None:
            # Validate node
            if isinstance(node, int):
                safe_node = node
            else:
                safe_node = validate_id(node, "Node")
                
            if safe_node not in graph:
                return {"error": f"Node '{safe_node}' not in graph"}
            
            if graph.is_directed():
                return {
                    "node": safe_node,
                    "in_degree": graph.in_degree(safe_node),
                    "out_degree": graph.out_degree(safe_node),
                    "total_degree": graph.degree(safe_node),
                }
            else:
                return {"node": safe_node, "degree": graph.degree(safe_node)}
        else:
            # Return degree for all nodes
            if graph.number_of_nodes() > 10000:
                return {"error": "Graph too large to return all degrees. Please query specific nodes."}
                
            if graph.is_directed():
                return {
                    "in_degrees": dict(graph.in_degree()),
                    "out_degrees": dict(graph.out_degree()),
                    "total_degrees": dict(graph.degree()),
                }
            else:
                return {"degrees": dict(graph.degree())}
                
    except ValidationError as e:
        logger.warning(f"Validation error in node_degree: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
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


# Additional algorithm tools using GraphAlgorithms

@mcp.tool(description="Find connected components in a graph")
@with_resource_limits
def connected_components(graph_name: str) -> dict[str, Any]:
    """Find connected components using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.connected_components(graph)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in connected_components: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error finding connected components: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Calculate centrality measures for a graph")
@with_resource_limits
def centrality_measures(
    graph_name: str, 
    measures: list[str] | None = None
) -> dict[str, Any]:
    """Calculate centrality measures using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        # Check operation feasibility
        check_operation_feasibility(graph, "betweenness_centrality")
        
        result = graph_algorithms.centrality_measures(graph, measures)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in centrality_measures: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except nx.PowerIterationFailedConvergence:
        return {"error": "Eigenvector centrality failed to converge"}
    except Exception as e:
        logger.error(f"Error calculating centrality: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Calculate clustering coefficients")
@with_resource_limits
def clustering_coefficients(graph_name: str) -> dict[str, Any]:
    """Calculate clustering coefficients using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.clustering_coefficients(graph)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in clustering_coefficients: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error calculating clustering: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Find minimum spanning tree")
@with_resource_limits
def minimum_spanning_tree(
    graph_name: str,
    weight: str = "weight",
    algorithm: str = "kruskal"
) -> dict[str, Any]:
    """Find minimum spanning tree using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.minimum_spanning_tree(graph, weight, algorithm)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in minimum_spanning_tree: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except ValueError as e:
        return {"error": str(e)}  # For directed graphs or invalid algorithm
    except Exception as e:
        logger.error(f"Error finding MST: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Calculate maximum flow between source and sink")
@with_resource_limits
def maximum_flow(
    graph_name: str,
    source: Any,
    sink: Any,
    capacity: str = "capacity"
) -> dict[str, Any]:
    """Calculate maximum flow using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        # Validate source and sink
        if isinstance(source, int):
            safe_source = source
        else:
            safe_source = validate_id(source, "Source node")
            
        if isinstance(sink, int):
            safe_sink = sink
        else:
            safe_sink = validate_id(sink, "Sink node")
        
        result = graph_algorithms.maximum_flow(graph, safe_source, safe_sink, capacity)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in maximum_flow: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except ValueError as e:
        return {"error": str(e)}  # For undirected graphs or missing nodes
    except nx.NetworkXError as e:
        return {"error": f"Flow calculation error: {str(e)}"}
    except Exception as e:
        logger.error(f"Error calculating maximum flow: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Color graph vertices using greedy algorithm")
@with_resource_limits
def graph_coloring(
    graph_name: str,
    strategy: str = "largest_first"
) -> dict[str, Any]:
    """Color graph vertices using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.graph_coloring(graph, strategy)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in graph_coloring: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error coloring graph: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Detect communities in a graph")
@with_resource_limits
def community_detection(
    graph_name: str,
    method: str = "louvain"
) -> dict[str, Any]:
    """Detect communities using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        # Check operation feasibility
        check_operation_feasibility(graph, "betweenness_centrality")
        
        result = graph_algorithms.community_detection(graph, method)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in community_detection: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except ImportError as e:
        return {"error": "Community detection algorithms not available in this NetworkX version"}
    except ValueError as e:
        return {"error": str(e)}  # For unknown method
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Detect cycles in a graph")
@with_resource_limits
def cycles_detection(graph_name: str) -> dict[str, Any]:
    """Detect cycles using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.cycles_detection(graph)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in cycles_detection: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error detecting cycles: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Find matching in a graph")
@with_resource_limits
def matching(
    graph_name: str,
    max_cardinality: bool = True
) -> dict[str, Any]:
    """Find matching using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.matching(graph, max_cardinality)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in matching: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error finding matching: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Calculate comprehensive graph statistics")
@with_resource_limits
def graph_statistics(graph_name: str) -> dict[str, Any]:
    """Calculate graph statistics using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        result = graph_algorithms.graph_statistics(graph)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in graph_statistics: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {"error": safe_error_message(e)}


@mcp.tool(description="Find all pairs shortest paths")
@with_resource_limits
def all_pairs_shortest_path(
    graph_name: str,
    weight: str | None = None
) -> dict[str, Any]:
    """Find all pairs shortest paths using GraphAlgorithms."""
    try:
        safe_graph_name = validate_id(graph_name, "Graph name")
        graph = graph_manager.get_graph(safe_graph_name)
        
        # Check operation feasibility - this is O(V^3)
        check_operation_feasibility(graph, "all_pairs_shortest_path")
        
        result = graph_algorithms.all_pairs_shortest_path(graph, weight)
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation error in all_pairs_shortest_path: {e}")
        return {"error": str(e)}
    except KeyError as e:
        return {"error": f"Graph '{graph_name}' not found"}
    except ResourceLimitError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error finding all pairs shortest paths: {e}")
        return {"error": safe_error_message(e)}


# Import feature flag management
from .features import get_flag_manager, get_feature_flags, set_feature_enabled

# Import monitoring endpoints
from .monitoring.endpoints import create_monitoring_endpoints


@mcp.tool(description="Manage feature flags (admin only)")
def manage_feature_flags(
    action: str = "list",
    flag_name: str | None = None,
    enabled: bool | None = None,
    admin_token: str | None = None
) -> dict[str, Any]:
    """
    Manage feature flags at runtime.
    
    Parameters:
    -----------
    action : str
        Action to perform: 'list', 'get', 'set', 'validate'
    flag_name : str
        Feature flag name (for get/set actions)
    enabled : bool
        Enable or disable flag (for set action)
    admin_token : str
        Admin authentication token
    
    Returns:
    --------
    Dict with feature flag information or operation result
    """
    # Simple admin check - in production, use proper authentication
    expected_token = os.getenv("FEATURE_FLAG_ADMIN_TOKEN", "admin-secret")
    if action != "list" and admin_token != expected_token:
        return {"error": "Unauthorized: Invalid admin token"}
    
    manager = get_flag_manager()
    
    if action == "list":
        # List all feature flags (safe operation)
        flags = get_feature_flags()
        
        # Group by category
        by_category = {}
        for name, info in flags.items():
            category = info.get("category", "general")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append({
                "name": name,
                "enabled": info["enabled"],
                "status": info["status"],
                "description": info["description"],
                "requires_restart": info.get("requires_restart", False)
            })
        
        return {
            "total_flags": len(flags),
            "by_category": by_category,
            "ml_enabled": manager.is_enabled("ml_base_features"),
            "experimental_allowed": os.getenv("MCP_ENVIRONMENT") != "production"
        }
    
    elif action == "get":
        # Get specific flag information
        if not flag_name:
            return {"error": "flag_name required for get action"}
        
        if flag_name not in manager.flags:
            return {"error": f"Unknown feature flag: {flag_name}"}
        
        flag = manager.flags[flag_name]
        return {
            "name": flag_name,
            "enabled": flag.is_enabled(),
            "status": flag.status.value,
            "description": flag.description,
            "category": flag.category,
            "tags": flag.tags,
            "requires_restart": flag.requires_restart,
            "dependencies": flag.dependencies,
            "conflicts": flag.conflicts,
            "rollout_percentage": flag.rollout_percentage,
            "updated_at": flag.updated_at.isoformat()
        }
    
    elif action == "set":
        # Set feature flag state
        if not flag_name:
            return {"error": "flag_name required for set action"}
        
        if enabled is None:
            return {"error": "enabled parameter required for set action"}
        
        success = set_feature_enabled(flag_name, enabled)
        
        if success:
            flag = manager.flags[flag_name]
            restart_msg = " (restart required for full effect)" if flag.requires_restart else ""
            return {
                "success": True,
                "flag_name": flag_name,
                "enabled": enabled,
                "message": f"Feature '{flag_name}' {'enabled' if enabled else 'disabled'}{restart_msg}"
            }
        else:
            return {"error": f"Failed to update feature flag: {flag_name}"}
    
    elif action == "validate":
        # Validate feature flag dependencies
        errors = manager.validate_dependencies()
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "total_flags": len(manager.flags)
        }
    
    else:
        return {"error": f"Unknown action: {action}. Use 'list', 'get', 'set', or 'validate'"}


# Initialize monitoring endpoints
monitoring_endpoints = create_monitoring_endpoints()


@mcp.tool(description="Health check endpoint")
def health_check(include_details: bool = True) -> dict[str, Any]:
    """
    Comprehensive health check for monitoring and load balancers.
    
    Parameters:
    -----------
    include_details : bool
        Whether to include detailed component information
        
    Returns:
    --------
    Dict containing health status and component checks
    """
    try:
        return monitoring_endpoints["health"].check_health(include_details)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": f"Health check system error: {str(e)}"
        }


@mcp.tool(description="Readiness check for Kubernetes")
def readiness_check() -> dict[str, Any]:
    """
    Kubernetes readiness probe - checks if service is ready for traffic.
    
    Returns:
    --------
    Dict indicating if service is ready to accept requests
    """
    try:
        return monitoring_endpoints["ready"].check_readiness()
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "ready": False,
            "timestamp": time.time(),
            "error": f"Readiness check system error: {str(e)}"
        }


@mcp.tool(description="Export Prometheus metrics")
def get_metrics(format: str = "prometheus") -> dict[str, Any]:
    """
    Export system and application metrics in Prometheus or JSON format.
    
    Parameters:
    -----------
    format : str
        Export format: 'prometheus' or 'json'
        
    Returns:
    --------
    Dict containing metrics data
    """
    try:
        metrics_data = monitoring_endpoints["metrics"].get_metrics(format)
        
        if format == "prometheus":
            return {
                "format": "prometheus",
                "content_type": "text/plain; version=0.0.4; charset=utf-8",
                "metrics": metrics_data
            }
        else:
            return {
                "format": "json",
                "content_type": "application/json",
                "metrics": json.loads(metrics_data) if isinstance(metrics_data, str) else metrics_data
            }
            
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        return {
            "error": f"Metrics export failed: {str(e)}",
            "format": format,
            "timestamp": time.time()
        }


@mcp.tool(description="Get performance statistics")
def get_performance_stats() -> dict[str, Any]:
    """
    Get detailed performance statistics for operations.
    
    Returns:
    --------
    Dict containing performance metrics and statistics
    """
    try:
        from .monitoring.performance_tracker import get_performance_tracker
        
        tracker = get_performance_tracker()
        
        return {
            "current_load": tracker.get_current_load(),
            "top_operations": tracker.get_top_operations(limit=10),
            "all_stats": tracker.get_all_stats(),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Performance stats failed: {e}")
        return {
            "error": f"Performance stats failed: {str(e)}",
            "timestamp": time.time()
        }


class NetworkXMCPServer:
    """Wrapper class for backward compatibility."""
    
    def __init__(self):
        self.mcp = mcp
        self.graphs = graphs
        self.graph_manager = graph_manager
        self.graph_algorithms = graph_algorithms
    
    def run(self):
        """Run the server."""
        return main()


def main():
    """Run the integrated MCP server."""
    with CorrelationContext(operation_name="server.startup"):
        logger.info("Starting NetworkX MCP Server", extra={
            "version": "integrated",
            "components": ["GraphManager", "GraphAlgorithms"],
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_format": os.getenv("LOG_FORMAT", "colored"),
            "log_file": os.getenv("LOG_FILE")
        })
        
        available_algorithms = [
            'shortest_path', 'connected_components', 'centrality_measures',
            'clustering_coefficients', 'minimum_spanning_tree', 'community_detection'
        ]
        
        logger.info("Server components initialized", extra={
            "available_algorithms": available_algorithms,
            "algorithm_count": len(available_algorithms),
            "monitoring_enabled": True,
            "feature_flags_enabled": True
        })
        
        try:
            # Run the server
            logger.info("Server starting - ready to accept requests")
            mcp.run()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested by user")
        except Exception as e:
            logger.error("Server startup failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            }, exc_info=True)
            raise


if __name__ == "__main__":
    main()