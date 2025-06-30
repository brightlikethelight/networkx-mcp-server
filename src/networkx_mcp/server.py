"""NetworkX MCP Server implementation using FastMCP."""

import base64
import json
import logging
import os
import random  # Using for non-cryptographic graph sampling only
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Union

import networkx as nx
import numpy as np

try:
    from networkx.algorithms.simple_paths import shortest_simple_paths
except ImportError:
    shortest_simple_paths = None

try:
    from fastmcp import FastMCP
    from mcp.types import TextContent

    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    try:
        from mcp import Server
        from mcp.server.models import InitializationOptions
        from mcp.types import TextContent
    except ImportError:
        # Use mock MCP when the package is not available
        from networkx_mcp.mcp_mock import MockMCP

        Server = MockMCP.Server
        FastMCP = MockMCP.FastMCP
        InitializationOptions = MockMCP.server.models.InitializationOptions
        TextContent = MockMCP.types.TextContent

# Phase 2 Advanced Analytics imports
from networkx_mcp.advanced import (
    BipartiteAnalysis,
    CommunityDetection,
    DirectedAnalysis,
    GraphGenerators,
    MLIntegration,
    NetworkFlow,
    RobustnessAnalysis,
    SpecializedAlgorithms,
)
from networkx_mcp.advanced.enterprise import EnterpriseFeatures
from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.core.io_handlers import GraphIOHandler
from networkx_mcp.integration import DataPipelines
from networkx_mcp.utils.formatters import GraphFormatter
from networkx_mcp.utils.monitoring import OperationCounter, PerformanceMonitor
from networkx_mcp.utils.validators import GraphValidator

# Phase 3 Visualization & Integration imports
from networkx_mcp.visualization import (
    MatplotlibVisualizer,
    PlotlyVisualizer,
    PyvisVisualizer,
    SpecializedVisualizations,
)

# Constants for edge and data validation
MIN_EDGE_ELEMENTS = 2
WEIGHTED_EDGE_ELEMENTS = 3

# Memory estimation constants
BYTES_PER_NODE = 100
BYTES_PER_EDGE = 50

# Performance and size thresholds
HISTOGRAM_DEFAULT_BINS = 10
MILLISECONDS_PER_SECOND = 1000
KILOBYTES_PER_MEGABYTE = 1024
MAX_ITERATION_DEFAULT = 1000
MAX_ITERATION_SMALL = 100
MAX_PATHS_DEFAULT = 100

# Analysis thresholds
MIN_NODES_FOR_DEGREE_ANALYSIS = 10
MIN_DEGREE_TYPES_FOR_POWER_LAW = 3
MAX_DEGREE_SAMPLES_FOR_POWER_LAW = 10
MIN_NODES_FOR_CONNECTIVITY = 2
TRIANGLE_DIVISION_FACTOR = 3
MAX_DISPLAY_ITEMS = 10

# Centrality calculation constants
IN_OUT_CENTRALITY_AVERAGE_FACTOR = 2

# Precision constants
DECIMAL_PLACES = 2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("networkx_mcp_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
if HAS_FASTMCP:
    mcp = FastMCP("NetworkX Graph Analysis Server")
else:
    mcp = Server("NetworkX Graph Analysis Server")
    mcp.request_context = InitializationOptions()

# Global state management
# For production deployments, use REDIS_URL environment variable to enable
# persistent state management via Redis backend (see storage.redis_backend)
graph_manager = GraphManager()
performance_monitor = PerformanceMonitor()
operation_counter = OperationCounter()
enterprise_features = EnterpriseFeatures()


@mcp.tool()
async def create_graph(
    graph_id: str,
    graph_type: str = "undirected",
    from_data: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create a new NetworkX graph with comprehensive initialization options.

    Args:
        graph_id: Unique identifier for the graph
        graph_type: Type of graph - 'undirected', 'directed', 'multigraph', 'multidigraph'
        from_data: Optional initialization data:
            - edge_list: List of edges [(source, target), ...] or [(source, target, weight), ...]
            - adjacency_matrix: 2D list representing adjacency matrix
            - node_labels: Optional labels for adjacency matrix nodes
        attributes: Additional graph attributes

    Returns:
        Graph creation status with metadata including:
        - graph_id: The created graph ID
        - graph_type: The type of graph created
        - num_nodes: Number of nodes
        - num_edges: Number of edges
        - created_at: Creation timestamp
        - initialization_method: How the graph was initialized
        - memory_estimate: Estimated memory usage
    """
    start_time = time.time()
    logger.info(f"Creating graph '{graph_id}' of type '{graph_type}'")

    try:
        # Map user-friendly types to NetworkX classes
        type_mapping = {
            "undirected": "Graph",
            "directed": "DiGraph",
            "multigraph": "MultiGraph",
            "multidigraph": "MultiDiGraph",
        }

        nx_graph_type = type_mapping.get(graph_type.lower(), graph_type)

        if not GraphValidator.validate_graph_type(nx_graph_type):
            error_msg = f"Invalid graph type: {graph_type}. Valid types: {list(type_mapping.keys())}"
            logger.error(error_msg)
            return GraphFormatter.format_error("ValidationError", error_msg)

        # Create the graph
        graph_attrs = attributes or {}
        result = graph_manager.create_graph(graph_id, nx_graph_type, **graph_attrs)

        # Initialize from data if provided
        if from_data:
            graph = graph_manager.get_graph(graph_id)
            initialization_method = "empty"

            if "edge_list" in from_data:
                edges = from_data["edge_list"]
                if edges and len(edges[0]) == WEIGHTED_EDGE_ELEMENTS:  # Weighted edges
                    graph.add_weighted_edges_from(edges)
                else:
                    graph.add_edges_from(edges)
                initialization_method = "edge_list"
                logger.info(f"Initialized graph from edge list with {len(edges)} edges")

            elif "adjacency_matrix" in from_data:
                matrix = np.array(from_data["adjacency_matrix"])
                node_labels = from_data.get("node_labels")

                if node_labels and len(node_labels) != matrix.shape[0]:
                    msg = "Node labels length must match matrix dimensions"
                    raise ValueError(msg)

                # Create graph from adjacency matrix
                temp_graph = nx.from_numpy_array(matrix, create_using=graph.__class__)

                # Relabel nodes if labels provided
                if node_labels:
                    mapping = dict(enumerate(node_labels))
                    temp_graph = nx.relabel_nodes(temp_graph, mapping)

                # Copy to our graph
                graph.add_nodes_from(temp_graph.nodes(data=True))
                graph.add_edges_from(temp_graph.edges(data=True))
                initialization_method = "adjacency_matrix"
                logger.info(f"Initialized graph from adjacency matrix {matrix.shape}")

            result["initialization_method"] = initialization_method

        # Calculate memory estimate
        graph = graph_manager.get_graph(graph_id)
        memory_estimate = (
            graph.number_of_nodes() * BYTES_PER_NODE  # Rough estimate per node
            + graph.number_of_edges() * 50  # 50 bytes per edge
        ) / 1024  # Convert to KB

        result["memory_estimate_kb"] = round(memory_estimate, DECIMAL_PLACES)
        result["num_nodes"] = graph.number_of_nodes()
        result["num_edges"] = graph.number_of_edges()

        # Update monitoring
        elapsed_time = time.time() - start_time
        performance_monitor.record_operation("create_graph", elapsed_time)
        operation_counter.increment("create_graph")

        logger.info(f"Graph '{graph_id}' created successfully in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "create_graph", "Graph created successfully", result
        )

    except Exception as e:
        logger.error(f"Error creating graph '{graph_id}': {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("GraphCreationError", str(e))


@mcp.tool()
async def delete_graph(graph_id: str) -> dict[str, Any]:
    """
    Delete a graph by ID.

    Args:
        graph_id: ID of the graph to delete

    Returns:
        Deletion status
    """
    try:
        result = graph_manager.delete_graph(graph_id)
        return GraphFormatter.format_success(
            "delete_graph", "Graph deleted successfully", result
        )
    except Exception as e:
        return GraphFormatter.format_error("GraphDeletionError", str(e))


@mcp.tool()
async def list_graphs() -> dict[str, Any]:
    """
    List all available graphs.

    Returns:
        List of graphs with their metadata
    """
    try:
        graphs = graph_manager.list_graphs()
        return GraphFormatter.format_success(
            "list_graphs", f"Found {len(graphs)} graphs", {"graphs": graphs}
        )
    except Exception as e:
        return GraphFormatter.format_error("ListGraphsError", str(e))


@mcp.tool()
async def get_graph_info(graph_id: str) -> dict[str, Any]:
    """
    Get comprehensive information about a graph.

    Args:
        graph_id: ID of the graph to analyze

    Returns:
        Detailed graph information including:
        - Basic properties: nodes, edges, density, type
        - Connectivity: is_connected, components count
        - Degree statistics: min, max, average, distribution
        - Memory usage estimation
        - Graph properties: has_self_loops, is_weighted, etc.
        - Performance characteristics
    """
    start_time = time.time()
    logger.info(f"Getting info for graph '{graph_id}'")

    try:
        # Get basic info
        info = graph_manager.get_graph_info(graph_id)
        graph = graph_manager.get_graph(graph_id)

        # Add connectivity information
        if graph.is_directed():
            info["is_weakly_connected"] = nx.is_weakly_connected(graph)
            info["is_strongly_connected"] = nx.is_strongly_connected(graph)
            info["num_weakly_connected_components"] = (
                nx.number_weakly_connected_components(graph)
            )
            info["num_strongly_connected_components"] = (
                nx.number_strongly_connected_components(graph)
            )
        else:
            info["is_connected"] = nx.is_connected(graph)
            info["num_connected_components"] = nx.number_connected_components(graph)

        # Add graph properties
        info["has_self_loops"] = any(u == v for u, v in graph.edges())
        info["is_weighted"] = any(
            "weight" in data for _, _, data in graph.edges(data=True)
        )

        # Memory usage estimation
        memory_estimate = (
            graph.number_of_nodes() * BYTES_PER_NODE  # bytes per node
            + graph.number_of_edges() * 50  # ~50 bytes per edge
            + len(str(graph.nodes(data=True)))  # Node attributes
            + len(str(graph.edges(data=True)))  # Edge attributes
        ) / 1024  # Convert to KB

        info["memory_usage_kb"] = round(memory_estimate, DECIMAL_PLACES)

        # Add degree distribution summary
        if graph.number_of_nodes() > 0:
            degrees = [d for n, d in graph.degree()]
            info["degree_distribution"] = {
                "values": np.histogram(
                    degrees, bins=min(HISTOGRAM_DEFAULT_BINS, len(set(degrees)))
                )[0].tolist(),
                "bins": np.histogram(
                    degrees, bins=min(HISTOGRAM_DEFAULT_BINS, len(set(degrees)))
                )[1].tolist(),
            }

        # Performance metrics
        elapsed_time = time.time() - start_time
        info["query_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        operation_counter.increment("get_graph_info")

        logger.info(f"Retrieved info for graph '{graph_id}' in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "get_graph_info", "Graph info retrieved", info
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error getting info for graph '{graph_id}': {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("GraphInfoError", str(e))


@mcp.tool()
async def add_nodes(
    graph_id: str,
    nodes: Union[list[str], list[dict[str, Any]]],
    node_attributes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Add nodes to a graph with support for bulk operations and attributes.

    Args:
        graph_id: ID of the graph to add nodes to
        nodes: Either:
            - List of node IDs: ['node1', 'node2', ...]
            - List of dicts with 'id' and attributes: [{'id': 'node1', 'color': 'red'}, ...]
        node_attributes: Optional default attributes to apply to all nodes

    Returns:
        Operation status including:
        - nodes_added: Number of nodes added
        - total_nodes: Total nodes in graph after operation
        - nodes_with_attributes: Number of nodes that have attributes
        - execution_time_ms: Time taken for the operation
    """
    start_time = time.time()
    logger.info(f"Adding {len(nodes)} nodes to graph '{graph_id}'")

    try:
        # Validate graph exists
        graph = graph_manager.get_graph(graph_id)
        initial_node_count = graph.number_of_nodes()

        # Process nodes with validation
        processed_nodes = []
        nodes_with_attrs = 0

        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                if "id" not in node:
                    msg = f"Node at index {i} is missing 'id' field"
                    raise ValueError(msg)

                node_id = node["id"]
                if not GraphValidator.validate_node_id(node_id):
                    msg = f"Invalid node ID at index {i}: {node_id}"
                    raise ValueError(msg)

                # Merge node-specific attributes with default attributes
                attrs = {k: v for k, v in node.items() if k != "id"}
                if node_attributes:
                    attrs = {
                        **node_attributes,
                        **attrs,
                    }  # Node-specific overrides defaults

                if attrs:
                    nodes_with_attrs += 1
                    processed_nodes.append((node_id, attrs))
                else:
                    processed_nodes.append(node_id)
            else:
                # Simple node ID
                if not GraphValidator.validate_node_id(node):
                    msg = f"Invalid node ID at index {i}: {node}"
                    raise ValueError(msg)

                if node_attributes:
                    processed_nodes.append((node, node_attributes))
                    nodes_with_attrs += 1
                else:
                    processed_nodes.append(node)

        # Bulk add nodes
        result = graph_manager.add_nodes_from(graph_id, processed_nodes)

        # Add detailed statistics
        result["nodes_with_attributes"] = nodes_with_attrs
        result["duplicate_nodes_skipped"] = len(nodes) - (
            result["total_nodes"] - initial_node_count
        )

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["execution_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("add_nodes", elapsed_time)
        operation_counter.increment("add_nodes", len(nodes))

        logger.info(
            f"Added {result['nodes_added']} nodes to graph '{graph_id}' in {elapsed_time:.3f}s"
        )
        return GraphFormatter.format_success(
            "add_nodes", "Nodes added successfully", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error adding nodes to graph '{graph_id}': {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("AddNodesError", str(e))


@mcp.tool()
async def add_edges(
    graph_id: str,
    edges: list[Union[tuple, dict[str, Any]]],
    edge_attributes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Add edges to a graph with support for bulk operations, weights, and attributes.

    Args:
        graph_id: ID of the graph to add edges to
        edges: List of edges in any of these formats:
            - Tuples: [(source, target), ...] or [(source, target, weight), ...]
            - Dicts: [{'source': 'A', 'target': 'B', 'weight': 1.5, ...}, ...]
        edge_attributes: Optional default attributes to apply to all edges

    Returns:
        Operation status including:
        - edges_added: Number of edges added
        - total_edges: Total edges in graph after operation
        - weighted_edges: Number of edges with weights
        - multi_edges_created: Number of multi-edges (for multigraphs)
        - self_loops_added: Number of self-loops added
        - execution_time_ms: Time taken for the operation
    """
    start_time = time.time()
    logger.info(f"Adding {len(edges)} edges to graph '{graph_id}'")

    try:
        # Validate graph exists
        graph = graph_manager.get_graph(graph_id)
        initial_edge_count = graph.number_of_edges()
        is_multigraph = graph.is_multigraph()

        # Process edges with validation
        processed_edges = []
        weighted_edges = 0
        self_loops = 0

        for i, edge in enumerate(edges):
            if isinstance(edge, dict):
                if "source" not in edge or "target" not in edge:
                    msg = f"Edge at index {i} missing 'source' or 'target'"
                    raise ValueError(msg)

                source, target = edge["source"], edge["target"]

                # Extract attributes
                attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
                if edge_attributes:
                    attrs = {
                        **edge_attributes,
                        **attrs,
                    }  # Edge-specific overrides defaults

                if "weight" in attrs:
                    weighted_edges += 1

                if source == target:
                    self_loops += 1

                processed_edges.append(
                    (source, target, attrs) if attrs else (source, target)
                )

            elif isinstance(edge, (tuple, list)):
                if len(edge) < MIN_EDGE_ELEMENTS:
                    msg = f"Edge at index {i} must have at least source and target"
                    raise ValueError(msg)

                source, target = edge[0], edge[1]

                if source == target:
                    self_loops += 1

                if len(edge) == WEIGHTED_EDGE_ELEMENTS:
                    # (source, target, weight) format
                    attrs = {"weight": edge[WEIGHTED_EDGE_ELEMENTS - 1]}
                    if edge_attributes:
                        attrs.update(edge_attributes)
                    weighted_edges += 1
                    processed_edges.append((source, target, attrs))
                elif len(edge) == MIN_EDGE_ELEMENTS and edge_attributes:
                    # Apply default attributes
                    if "weight" in edge_attributes:
                        weighted_edges += 1
                    processed_edges.append((source, target, edge_attributes))
                else:
                    processed_edges.append((source, target))
            else:
                msg = f"Invalid edge format at index {i}"
                raise ValueError(msg)

        # Validate nodes exist
        for edge in processed_edges:
            source, target = edge[0], edge[1]
            if source not in graph:
                logger.warning(f"Source node '{source}' not in graph, will be created")
            if target not in graph:
                logger.warning(f"Target node '{target}' not in graph, will be created")

        # Bulk add edges
        result = graph_manager.add_edges_from(graph_id, processed_edges)

        # Calculate multi-edges if applicable
        multi_edges = 0
        if is_multigraph:
            for u, v in graph.edges():
                if graph.number_of_edges(u, v) > 1:
                    multi_edges += graph.number_of_edges(u, v) - 1

        # Add detailed statistics
        result["weighted_edges"] = weighted_edges
        result["self_loops_added"] = self_loops
        result["multi_edges_created"] = multi_edges
        result["duplicate_edges_skipped"] = len(edges) - (
            result["total_edges"] - initial_edge_count
        )

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["execution_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("add_edges", elapsed_time)
        operation_counter.increment("add_edges", len(edges))

        logger.info(
            f"Added {result['edges_added']} edges to graph '{graph_id}' in {elapsed_time:.3f}s"
        )
        return GraphFormatter.format_success(
            "add_edges", "Edges added successfully", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error adding edges to graph '{graph_id}': {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("AddEdgesError", str(e))


@mcp.tool()
async def shortest_path(
    graph_id: str,
    source: Union[str, int],
    target: Optional[Union[str, int]] = None,
    weight: Optional[str] = None,
    method: str = "dijkstra",
    k_paths: Optional[int] = None,
) -> dict[str, Any]:
    """
    Find shortest path(s) in a graph with multiple algorithm support.

    Args:
        graph_id: ID of the graph
        source: Source node
        target: Target node (optional - if not provided, finds paths to all nodes)
        weight: Edge weight attribute name (None for unweighted)
        method: Algorithm - 'dijkstra', 'bellman-ford', 'floyd-warshall', 'astar'
        k_paths: Number of shortest paths to find (for k-shortest paths)

    Returns:
        Path information including:
        - path(s): Node sequence(s)
        - distance(s): Total path weight(s)
        - algorithm_used: Which algorithm was applied
        - computation_time: Time taken
        - path_count: Number of paths found
    """
    start_time = time.time()
    logger.info(f"Finding shortest path in graph '{graph_id}' from '{source}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        # Validate inputs
        validation = GraphValidator.validate_algorithm_input(
            method, graph, {"source": source, "target": target, "weight": weight}
        )
        if not validation["valid"]:
            return GraphFormatter.format_error(
                "ValidationError",
                "Invalid input for shortest path",
                {"errors": validation["errors"]},
            )

        # Check if graph is empty
        if graph.number_of_nodes() == 0:
            return GraphFormatter.format_error(
                "EmptyGraphError", "Cannot find paths in empty graph"
            )

        # Handle different algorithms
        result = {}

        if method == "floyd-warshall":
            # All pairs shortest paths
            if weight:
                pred, dist = nx.floyd_warshall_predecessor_and_distance(
                    graph, weight=weight
                )
            else:
                pred, dist = nx.floyd_warshall_predecessor_and_distance(graph)

            # Convert to serializable format
            result["all_pairs_distances"] = {
                str(u): {str(v): d for v, d in dists.items()}
                for u, dists in dist.items()
            }

            if target:
                # Extract specific path
                if source in dist and target in dist[source]:
                    path = nx.reconstruct_path(source, target, pred)
                    result["path"] = path
                    result["distance"] = dist[source][target]
                else:
                    result["path"] = None
                    result["distance"] = float("inf")

            result["algorithm"] = "floyd-warshall"

        elif k_paths and k_paths > 1 and target:
            # K-shortest paths
            try:
                if shortest_simple_paths is None:
                    result["error"] = "shortest_simple_paths not available"
                    return result

                k_shortest = []
                path_gen = shortest_simple_paths(graph, source, target, weight=weight)

                for i, path in enumerate(path_gen):
                    if i >= k_paths:
                        break

                    # Calculate path length
                    if weight:
                        length = sum(
                            graph[u][v].get(weight, 1)
                            for u, v in zip(path[:-1], path[1:])
                        )
                    else:
                        length = len(path) - 1

                    k_shortest.append({"path": path, "distance": length, "rank": i + 1})

                result["k_shortest_paths"] = k_shortest
                result["algorithm"] = "k_shortest_paths"

            except Exception as e:
                logger.warning(
                    f"K-shortest paths failed: {e}, falling back to single path"
                )
                k_paths = 1

        if not result:  # Standard single shortest path
            sp_result = GraphAlgorithms.shortest_path(
                graph, source, target, weight, method
            )

            if target:
                result = {
                    "path": sp_result.get("path"),
                    "distance": sp_result.get("length", sp_result.get("distance")),
                    "algorithm": method,
                }
            else:
                # All shortest paths from source
                result = {
                    "paths": sp_result.get("paths", {}),
                    "distances": sp_result.get(
                        "lengths", sp_result.get("distances", {})
                    ),
                    "algorithm": method,
                    "reachable_nodes": len(sp_result.get("paths", {})),
                }

        # Add performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        # Handle negative cycles for Bellman-Ford
        if method == "bellman-ford":
            try:
                negative_cycle = nx.negative_edge_cycle(graph, weight=weight)
                result["has_negative_cycle"] = True
                result["negative_cycle"] = list(negative_cycle)
            except nx.NetworkXError:
                result["has_negative_cycle"] = False

        performance_monitor.record_operation("shortest_path", elapsed_time)
        operation_counter.increment("shortest_path")

        logger.info(f"Shortest path computation completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "shortest_path", "Path computation successful", result
        )

    except nx.NetworkXNoPath:
        return GraphFormatter.format_error(
            "NoPathError", f"No path exists from '{source}' to '{target}'"
        )
    except nx.NodeNotFound as e:
        return GraphFormatter.format_error("NodeNotFoundError", str(e))
    except Exception as e:
        logger.error(f"Error in shortest path: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("ShortestPathError", str(e))


@mcp.tool()
async def calculate_centrality(
    graph_id: str,
    centrality_type: Union[str, list[str]] = "degree",
    top_n: Optional[int] = MAX_DISPLAY_ITEMS,
    include_statistics: bool = True,
    weight: Optional[str] = None,
) -> dict[str, Any]:
    """
    Calculate various centrality measures for nodes in a graph.

    Args:
        graph_id: ID of the graph
        centrality_type: Type(s) of centrality - 'degree', 'betweenness', 'closeness',
                        'eigenvector', 'pagerank', 'katz', 'harmonic'
        top_n: Number of top central nodes to highlight
        include_statistics: Whether to include mean, std, min, max statistics
        weight: Edge attribute to use as weight (for weighted centralities)

    Returns:
        Centrality results including:
        - centrality_scores: Dict of node -> centrality value
        - top_nodes: List of top N most central nodes
        - statistics: Mean, std, min, max of centrality values
        - distribution: Histogram of centrality values
        - computation_time: Time taken for calculation
    """
    start_time = time.time()
    logger.info(f"Calculating centrality for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        # Handle empty graph
        if graph.number_of_nodes() == 0:
            return GraphFormatter.format_error(
                "EmptyGraphError", "Cannot calculate centrality for empty graph"
            )

        # Normalize centrality_type to list
        if isinstance(centrality_type, str):
            centrality_types = [centrality_type]
        else:
            centrality_types = centrality_type

        # Validate centrality types
        for ctype in centrality_types:
            if not GraphValidator.validate_centrality_measure(ctype):
                return GraphFormatter.format_error(
                    "ValidationError",
                    f"Invalid centrality type: {ctype}. Valid types: degree, betweenness, closeness, eigenvector, pagerank, katz, harmonic",
                )

        results = {}

        for ctype in centrality_types:
            logger.info(f"Calculating {ctype} centrality")
            centrality_data = {}

            try:
                # Calculate centrality based on type
                if ctype == "degree":
                    if graph.is_directed():
                        in_cent = nx.in_degree_centrality(graph)
                        out_cent = nx.out_degree_centrality(graph)
                        centrality = {
                            n: (in_cent[n] + out_cent[n])
                            / IN_OUT_CENTRALITY_AVERAGE_FACTOR
                            for n in graph.nodes()
                        }
                        centrality_data["in_degree_centrality"] = in_cent
                        centrality_data["out_degree_centrality"] = out_cent
                    else:
                        centrality = nx.degree_centrality(graph)

                elif ctype == "betweenness":
                    centrality = nx.betweenness_centrality(
                        graph, weight=weight, normalized=True, endpoints=False
                    )

                elif ctype == "closeness":
                    centrality = nx.closeness_centrality(graph, distance=weight)

                elif ctype == "eigenvector":
                    if graph.number_of_edges() == 0:
                        centrality = dict.fromkeys(graph.nodes(), 0.0)
                    else:
                        try:
                            centrality = nx.eigenvector_centrality(
                                graph,
                                weight=weight,
                                max_iter=MAX_ITERATION_DEFAULT,
                                tol=1e-06,
                            )
                        except nx.PowerIterationFailedConvergence:
                            # Fallback to numpy method
                            centrality = nx.eigenvector_centrality_numpy(
                                graph, weight=weight
                            )

                elif ctype == "pagerank":
                    centrality = nx.pagerank(
                        graph, weight=weight, alpha=0.85, max_iter=MAX_ITERATION_SMALL
                    )

                elif ctype == "katz":
                    try:
                        centrality = nx.katz_centrality(
                            graph, weight=weight, normalized=True
                        )
                    except (nx.PowerIterationFailedConvergence, ValueError) as e:
                        # Fallback with smaller alpha
                        logger.debug(
                            f"Katz centrality failed with default alpha, using smaller value: {e}"
                        )
                        centrality = nx.katz_centrality(
                            graph, alpha=0.01, weight=weight, normalized=True
                        )

                elif ctype == "harmonic":
                    centrality = nx.harmonic_centrality(graph, distance=weight)

                # Store main centrality scores
                centrality_data["scores"] = centrality

                # Get top N nodes
                sorted_nodes = sorted(
                    centrality.items(), key=lambda x: x[1], reverse=True
                )

                if top_n:
                    centrality_data["top_nodes"] = [
                        {"node": node, "centrality": score, "rank": i + 1}
                        for i, (node, score) in enumerate(sorted_nodes[:top_n])
                    ]

                # Calculate statistics if requested
                if include_statistics and centrality:
                    values = list(centrality.values())
                    centrality_data["statistics"] = {
                        "mean": round(np.mean(values), 6),
                        "std": round(np.std(values), 6),
                        "min": round(min(values), 6),
                        "max": round(max(values), 6),
                        "median": round(np.median(values), 6),
                    }

                    # Create distribution histogram
                    hist, bins = np.histogram(values, bins=min(20, len(set(values))))
                    centrality_data["distribution"] = {
                        "counts": hist.tolist(),
                        "bins": [round(b, 6) for b in bins.tolist()],
                    }

                results[ctype] = centrality_data

            except Exception as e:
                logger.error(f"Error calculating {ctype} centrality: {e}")
                results[ctype] = {"error": str(e)}

        # Performance metrics
        elapsed_time = time.time() - start_time

        final_result = {
            "centrality_results": results,
            "graph_info": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "is_directed": graph.is_directed(),
            },
            "computation_time_ms": round(
                elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
            ),
        }

        performance_monitor.record_operation("calculate_centrality", elapsed_time)
        operation_counter.increment("calculate_centrality", len(centrality_types))

        logger.info(f"Centrality calculation completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "calculate_centrality", "Centrality calculated successfully", final_result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error calculating centrality: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("CentralityError", str(e))


@mcp.tool()
async def community_detection(graph_id: str, method: str = "louvain") -> dict[str, Any]:
    """
    Detect communities in a graph.

    Args:
        graph_id: ID of the graph
        method: Community detection algorithm

    Returns:
        Detected communities
    """
    try:
        graph = graph_manager.get_graph(graph_id)
        result = GraphAlgorithms.community_detection(graph, method)

        formatted_result = GraphFormatter.format_community_results(
            result["communities"], result.get("modularity")
        )

        return GraphFormatter.format_algorithm_result(
            "community_detection", graph_id=graph_id, method=method, **formatted_result
        )
    except Exception as e:
        return GraphFormatter.format_error("CommunityDetectionError", str(e))


@mcp.tool()
async def graph_statistics(graph_id: str) -> dict[str, Any]:
    """
    Calculate comprehensive graph statistics.

    Args:
        graph_id: ID of the graph

    Returns:
        Various graph statistics
    """
    try:
        graph = graph_manager.get_graph(graph_id)
        stats = GraphAlgorithms.graph_statistics(graph)

        return GraphFormatter.format_algorithm_result(
            "graph_statistics", graph_id=graph_id, statistics=stats
        )
    except Exception as e:
        return GraphFormatter.format_error("StatisticsError", str(e))


@mcp.tool()
async def export_graph(
    graph_id: str,
    format: str,
    path: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Export a graph to various formats.

    Args:
        graph_id: ID of the graph
        format: Export format (json, graphml, gexf, etc.)
        path: Output file path (optional for some formats)
        options: Format-specific options

    Returns:
        Export status or data
    """
    try:
        if not GraphValidator.validate_file_format(format, "export"):
            return GraphFormatter.format_error(
                "ValidationError", f"Unsupported export format: {format}"
            )

        graph = graph_manager.get_graph(graph_id)
        export_options = options or {}
        result = GraphIOHandler.export_graph(graph, format, path, **export_options)

        return GraphFormatter.format_success(
            "export_graph",
            f"Graph exported to {format} format",
            {"format": format, "result": result},
        )
    except Exception as e:
        return GraphFormatter.format_error("ExportError", str(e))


@mcp.tool()
async def import_graph(
    graph_id: str,
    format: str,
    data: Optional[dict[str, Any]] = None,
    path: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Import a graph from various formats.

    Args:
        graph_id: ID for the imported graph
        format: Import format (json, graphml, gexf, etc.)
        data: Graph data (for formats that support direct data)
        path: Input file path (for file-based formats)
        options: Format-specific options

    Returns:
        Import status
    """
    try:
        if not GraphValidator.validate_file_format(format, "import"):
            return GraphFormatter.format_error(
                "ValidationError", f"Unsupported import format: {format}"
            )

        # Import the graph
        import_options = options or {}
        graph = GraphIOHandler.import_graph(format, data, path, **import_options)

        # Add to graph manager
        graph_type = type(graph).__name__
        graph_manager.graphs[graph_id] = graph
        graph_manager.metadata[graph_id] = {
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "graph_type": graph_type,
            "imported_from": format,
        }

        info = graph_manager.get_graph_info(graph_id)

        return GraphFormatter.format_success(
            "import_graph", f"Graph imported from {format} format", info
        )
    except Exception as e:
        return GraphFormatter.format_error("ImportError", str(e))


@mcp.tool()
async def visualize_graph_simple(
    graph_id: str,
    layout: str = "spring",
    output_format: str = "pyvis",
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate visualization data for a graph.

    Args:
        graph_id: ID of the graph
        layout: Layout algorithm
        output_format: Visualization format (pyvis, plotly, matplotlib)
        options: Visualization options

    Returns:
        Visualization data or file path
    """
    try:
        graph = graph_manager.get_graph(graph_id)

        # Calculate layout
        layout_options = options or {}
        if layout == "spring":
            pos = nx.spring_layout(graph, **layout_options)
        elif layout == "circular":
            pos = nx.circular_layout(graph, **layout_options)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph, **layout_options)
        else:
            pos = nx.random_layout(graph, **layout_options)

        # Prepare node and edge data
        nodes = []
        for node in graph.nodes():
            nodes.append({"id": node, "attributes": dict(graph.nodes[node])})

        edges = []
        for source, target, attrs in graph.edges(data=True):
            edges.append({"source": source, "target": target, "attributes": attrs})

        # Convert positions to serializable format
        layout_data = {str(node): [float(x), float(y)] for node, (x, y) in pos.items()}

        result = GraphFormatter.format_visualization_data(
            nodes,
            edges,
            layout_data,
            layout_algorithm=layout,
            output_format=output_format,
        )

        return GraphFormatter.format_success(
            "visualize_graph", "Visualization data generated", result
        )
    except Exception as e:
        return GraphFormatter.format_error("VisualizationError", str(e))


@mcp.tool()
async def graph_metrics(
    graph_id: str, include_distributions: bool = True
) -> dict[str, Any]:
    """
    Calculate comprehensive graph metrics and statistics.

    Args:
        graph_id: ID of the graph to analyze
        include_distributions: Whether to include degree/distance distributions

    Returns:
        Comprehensive metrics including:
        - Basic metrics: density, sparsity, order, size
        - Degree statistics: distribution, assortativity, power law analysis
        - Distance metrics: diameter, radius, average path length
        - Connectivity: components, articulation points, bridges
        - Structural metrics: clustering, transitivity, reciprocity
        - Performance metrics for analysis
    """
    start_time = time.time()
    logger.info(f"Calculating comprehensive metrics for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)
        metrics = {}

        # Basic metrics
        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        metrics["basic"] = {
            "order": n,  # Number of nodes
            "size": m,  # Number of edges
            "density": nx.density(graph),
            "sparsity": 1 - nx.density(graph),
            "is_directed": graph.is_directed(),
            "is_multigraph": graph.is_multigraph(),
            "has_self_loops": any(u == v for u, v in graph.edges()),
            "memory_estimate_kb": round(
                (n * BYTES_PER_NODE + m * BYTES_PER_EDGE) / KILOBYTES_PER_MEGABYTE,
                DECIMAL_PLACES,
            ),
        }

        # Degree statistics
        degrees = dict(graph.degree())
        degree_values = list(degrees.values())

        if n > 0:
            metrics["degree"] = {
                "average": round(sum(degree_values) / n, DECIMAL_PLACES),
                "min": min(degree_values),
                "max": max(degree_values),
                "std": round(np.std(degree_values), DECIMAL_PLACES),
                "median": np.median(degree_values),
            }

            # Degree distribution
            if include_distributions:
                hist, bins = np.histogram(
                    degree_values, bins=min(20, len(set(degree_values)))
                )
                metrics["degree"]["distribution"] = {
                    "counts": hist.tolist(),
                    "bins": bins.tolist(),
                }

            # Assortativity
            try:
                metrics["degree"]["assortativity"] = round(
                    nx.degree_assortativity_coefficient(graph), 4
                )
            except (ValueError, nx.NetworkXError) as e:
                logger.debug(f"Failed to compute degree assortativity: {e}")
                metrics["degree"]["assortativity"] = None

            # Power law analysis
            if n > MIN_NODES_FOR_DEGREE_ANALYSIS:  # Need enough nodes
                unique_degrees = sorted(set(degree_values), reverse=True)
                degree_counts = [degree_values.count(d) for d in unique_degrees]

                # Simple power law check: log-log linearity
                if len(unique_degrees) > MIN_DEGREE_TYPES_FOR_POWER_LAW:
                    log_degrees = np.log(
                        unique_degrees[:MAX_DEGREE_SAMPLES_FOR_POWER_LAW]
                        if len(unique_degrees) > MAX_DEGREE_SAMPLES_FOR_POWER_LAW
                        else unique_degrees
                    )
                    log_counts = np.log(
                        degree_counts[:MAX_DEGREE_SAMPLES_FOR_POWER_LAW]
                        if len(degree_counts) > MAX_DEGREE_SAMPLES_FOR_POWER_LAW
                        else degree_counts
                    )

                    # Linear regression in log-log space
                    if len(log_degrees) > 1:
                        slope, intercept = np.polyfit(log_degrees, log_counts, 1)
                        metrics["degree"]["power_law_exponent"] = round(
                            -slope, DECIMAL_PLACES
                        )

        # For directed graphs
        if graph.is_directed():
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())

            metrics["in_degree"] = {
                "average": (
                    round(sum(in_degrees.values()) / n, DECIMAL_PLACES) if n > 0 else 0
                ),
                "min": min(in_degrees.values()) if in_degrees else 0,
                "max": max(in_degrees.values()) if in_degrees else 0,
            }

            metrics["out_degree"] = {
                "average": (
                    round(sum(out_degrees.values()) / n, DECIMAL_PLACES) if n > 0 else 0
                ),
                "min": min(out_degrees.values()) if out_degrees else 0,
                "max": max(out_degrees.values()) if out_degrees else 0,
            }

            # Reciprocity
            metrics["reciprocity"] = round(nx.reciprocity(graph), 4) if m > 0 else 0

        # Distance metrics (only for connected graphs)
        if n > 1:
            if graph.is_directed():
                is_connected = nx.is_strongly_connected(graph)
            else:
                is_connected = nx.is_connected(graph)

            if is_connected:
                metrics["distance"] = {
                    "diameter": nx.diameter(graph),
                    "radius": nx.radius(graph),
                    "average_shortest_path_length": round(
                        nx.average_shortest_path_length(graph), DECIMAL_PLACES
                    ),
                }

                # Eccentricity statistics
                eccentricities = nx.eccentricity(graph)
                metrics["distance"]["eccentricity"] = {
                    "min": min(eccentricities.values()),
                    "max": max(eccentricities.values()),
                    "average": round(
                        sum(eccentricities.values()) / len(eccentricities),
                        DECIMAL_PLACES,
                    ),
                }
            else:
                metrics["distance"] = {"connected": False}

        # Connectivity metrics
        if graph.is_directed():
            metrics["connectivity"] = {
                "is_weakly_connected": nx.is_weakly_connected(graph),
                "is_strongly_connected": nx.is_strongly_connected(graph),
                "num_weakly_connected_components": nx.number_weakly_connected_components(
                    graph
                ),
                "num_strongly_connected_components": nx.number_strongly_connected_components(
                    graph
                ),
            }
        else:
            metrics["connectivity"] = {
                "is_connected": nx.is_connected(graph),
                "num_connected_components": nx.number_connected_components(graph),
            }

            # Additional connectivity metrics for undirected
            if n > MIN_NODES_FOR_CONNECTIVITY:
                try:
                    metrics["connectivity"]["node_connectivity"] = nx.node_connectivity(
                        graph
                    )
                    metrics["connectivity"]["edge_connectivity"] = nx.edge_connectivity(
                        graph
                    )
                except Exception as e:
                    logger.debug(f"Failed to compute connectivity metrics: {e}")

                # Articulation points and bridges
                try:
                    articulation_points = list(nx.articulation_points(graph))
                    bridges = list(nx.bridges(graph))

                    metrics["connectivity"]["num_articulation_points"] = len(
                        articulation_points
                    )
                    metrics["connectivity"]["num_bridges"] = len(bridges)

                    if len(articulation_points) <= MAX_DISPLAY_ITEMS:
                        metrics["connectivity"][
                            "articulation_points"
                        ] = articulation_points
                    if len(bridges) <= MAX_DISPLAY_ITEMS:
                        metrics["connectivity"]["bridges"] = bridges
                except Exception as e:
                    logger.debug(f"Failed to compute articulation points/bridges: {e}")

        # Clustering and triangles
        if not graph.is_directed() or nx.is_directed_acyclic_graph(graph):
            clustering_coeffs = nx.clustering(graph)
            metrics["clustering"] = {
                "average_clustering": round(nx.average_clustering(graph), 4),
                "transitivity": round(nx.transitivity(graph), 4),
                "min_clustering": (
                    round(min(clustering_coeffs.values()), 4)
                    if clustering_coeffs
                    else 0
                ),
                "max_clustering": (
                    round(max(clustering_coeffs.values()), 4)
                    if clustering_coeffs
                    else 0
                ),
            }

            # Triangle count
            try:
                triangles = (
                    sum(nx.triangles(graph).values()) // TRIANGLE_DIVISION_FACTOR
                )
                metrics["clustering"]["num_triangles"] = triangles
            except Exception as e:
                logger.debug(f"Failed to compute triangle count: {e}")

        # Performance metrics
        elapsed_time = time.time() - start_time
        metrics["performance"] = {
            "calculation_time_ms": round(
                elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
            ),
            "metrics_calculated": len(metrics),
        }

        performance_monitor.record_operation("graph_metrics", elapsed_time)
        operation_counter.increment("graph_metrics")

        logger.info(f"Calculated metrics for graph '{graph_id}' in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "graph_metrics", "Metrics calculated successfully", metrics
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error calculating metrics for graph '{graph_id}': {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("MetricsError", str(e))


@mcp.tool()
async def monitoring_stats() -> dict[str, Any]:
    """
    Get server monitoring statistics and performance metrics.

    Returns:
        Server statistics including:
        - Operation counts and distribution
        - Performance metrics (avg, min, max times)
        - Error rates by operation
        - Memory usage estimates
        - Uptime and hourly patterns
    """
    try:
        # Get operation statistics
        operation_stats = operation_counter.get_counts()

        # Get performance statistics
        performance_stats = performance_monitor.get_statistics()

        # Get memory usage for all graphs
        total_memory_kb = 0
        graph_memories = {}

        for graph_info in graph_manager.list_graphs():
            graph_id = graph_info["graph_id"]
            graph = graph_manager.get_graph(graph_id)
            memory_est = (
                graph.number_of_nodes() * BYTES_PER_NODE
                + graph.number_of_edges() * BYTES_PER_EDGE
            ) / KILOBYTES_PER_MEGABYTE
            graph_memories[graph_id] = round(memory_est, DECIMAL_PLACES)
            total_memory_kb += memory_est

        # Combine all statistics
        stats = {
            "server_info": {
                "name": "NetworkX Graph Analysis Server",
                "version": "1.0.0",
                "uptime": operation_stats["uptime"],
                "total_graphs": len(graph_manager.graphs),
            },
            "operations": operation_stats,
            "performance": performance_stats,
            "memory": {
                "total_memory_kb": round(total_memory_kb, DECIMAL_PLACES),
                "total_memory_mb": round(
                    total_memory_kb / KILOBYTES_PER_MEGABYTE, DECIMAL_PLACES
                ),
                "graphs": graph_memories,
            },
            "slow_operations": performance_monitor.get_slow_operations(
                threshold_ms=500
            ),
        }

        logger.info("Retrieved monitoring statistics")
        return GraphFormatter.format_success(
            "monitoring_stats", "Statistics retrieved", stats
        )

    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e!s}")
        return GraphFormatter.format_error("MonitoringError", str(e))


@mcp.tool()
async def clustering_analysis(
    graph_id: str,
    include_triangles: bool = True,
    nodes: Optional[list[Union[str, int]]] = None,
) -> dict[str, Any]:
    """
    Analyze clustering coefficients and triangles in a graph.

    Args:
        graph_id: ID of the graph
        include_triangles: Whether to count triangles
        nodes: Specific nodes to analyze (None for all nodes)

    Returns:
        Clustering analysis including:
        - local_clustering: Clustering coefficient for each node
        - global_clustering: Average clustering coefficient
        - transitivity: Global clustering coefficient (fraction of triangles)
        - triangle_count: Number of triangles (if requested)
        - node_triangles: Triangles per node
        - clustering_distribution: Distribution of clustering values
    """
    start_time = time.time()
    logger.info(f"Analyzing clustering for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        # For directed graphs, convert to undirected for clustering
        if graph.is_directed():
            logger.info(
                "Converting directed graph to undirected for clustering analysis"
            )
            graph_undirected = graph.to_undirected()
        else:
            graph_undirected = graph

        result = {}

        # Local clustering coefficients
        if nodes:
            # Validate nodes exist
            for node in nodes:
                if node not in graph:
                    return GraphFormatter.format_error(
                        "NodeNotFoundError", f"Node '{node}' not found in graph"
                    )
            local_clustering = nx.clustering(graph_undirected, nodes)
        else:
            local_clustering = nx.clustering(graph_undirected)

        result["local_clustering"] = local_clustering

        # Global metrics
        result["average_clustering"] = nx.average_clustering(graph_undirected)
        result["transitivity"] = nx.transitivity(graph_undirected)

        # Clustering statistics
        clustering_values = list(local_clustering.values())
        if clustering_values:
            result["clustering_statistics"] = {
                "mean": round(np.mean(clustering_values), 4),
                "std": round(np.std(clustering_values), 4),
                "min": round(min(clustering_values), 4),
                "max": round(max(clustering_values), 4),
                "num_nodes_with_clustering_1": sum(
                    1 for c in clustering_values if c == 1.0
                ),
                "num_nodes_with_clustering_0": sum(
                    1 for c in clustering_values if c == 0.0
                ),
            }

            # Distribution
            hist, bins = np.histogram(clustering_values, bins=HISTOGRAM_DEFAULT_BINS)
            result["clustering_distribution"] = {
                "counts": hist.tolist(),
                "bins": [round(b, 4) for b in bins.tolist()],
            }

        # Triangle analysis
        if include_triangles:
            triangles = nx.triangles(graph_undirected)
            result["node_triangles"] = triangles
            result["total_triangles"] = (
                sum(triangles.values()) // TRIANGLE_DIVISION_FACTOR
            )

            # Find nodes with most triangles
            sorted_triangles = sorted(
                triangles.items(), key=lambda x: x[1], reverse=True
            )
            result["top_triangle_nodes"] = [
                {"node": node, "triangles": count, "rank": i + 1}
                for i, (node, count) in enumerate(sorted_triangles[:MAX_DISPLAY_ITEMS])
            ]

        # For directed graphs, add directed-specific metrics
        if graph.is_directed():
            result["directed_metrics"] = {
                "reciprocity": nx.reciprocity(graph),
                "is_directed": True,
            }

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("clustering_analysis", elapsed_time)
        operation_counter.increment("clustering_analysis")

        logger.info(f"Clustering analysis completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "clustering_analysis", "Analysis completed successfully", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("ClusteringError", str(e))


@mcp.tool()
async def connected_components(
    graph_id: str,
    component_type: str = "weakly",
    return_sizes: bool = True,
    largest_only: bool = False,
) -> dict[str, Any]:
    """
    Find connected components in a graph.

    Args:
        graph_id: ID of the graph
        component_type: For directed graphs - 'weakly' or 'strongly' connected
        return_sizes: Whether to return component size distribution
        largest_only: Return only the largest component

    Returns:
        Component analysis including:
        - components: List of components (each component is a list of nodes)
        - num_components: Total number of components
        - component_sizes: Size of each component
        - largest_component_size: Size of the largest component
        - is_connected: Whether graph is fully connected
        - size_distribution: Distribution of component sizes
    """
    start_time = time.time()
    logger.info(f"Finding connected components in graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        result = {}

        if graph.is_directed():
            # Directed graph components
            if component_type.lower() == "strongly":
                components = list(nx.strongly_connected_components(graph))
                result["component_type"] = "strongly_connected"
                result["is_strongly_connected"] = nx.is_strongly_connected(graph)
            else:
                components = list(nx.weakly_connected_components(graph))
                result["component_type"] = "weakly_connected"
                result["is_weakly_connected"] = nx.is_weakly_connected(graph)

            # Also calculate the other type for comparison
            strong_components = list(nx.strongly_connected_components(graph))
            weak_components = list(nx.weakly_connected_components(graph))

            result["comparison"] = {
                "num_strongly_connected": len(strong_components),
                "num_weakly_connected": len(weak_components),
            }
        else:
            # Undirected graph components
            components = list(nx.connected_components(graph))
            result["component_type"] = "connected"
            result["is_connected"] = nx.is_connected(graph)

        # Sort components by size (largest first)
        components = sorted(components, key=len, reverse=True)

        if largest_only and components:
            # Return only the largest component
            largest = components[0]
            result["largest_component"] = {
                "nodes": list(largest),
                "size": len(largest),
                "fraction_of_graph": (
                    len(largest) / graph.number_of_nodes()
                    if graph.number_of_nodes() > 0
                    else 0
                ),
            }
        else:
            # Return all components
            result["components"] = [list(comp) for comp in components]
            result["num_components"] = len(components)

        # Component sizes
        if return_sizes and components:
            sizes = [len(comp) for comp in components]
            result["component_sizes"] = sizes
            result["largest_component_size"] = max(sizes) if sizes else 0
            result["smallest_component_size"] = min(sizes) if sizes else 0

            # Size distribution
            result["size_statistics"] = {
                "mean": round(np.mean(sizes), DECIMAL_PLACES),
                "std": round(np.std(sizes), DECIMAL_PLACES),
                "median": np.median(sizes),
            }

            # Component size distribution
            size_counts = {}
            for size in sizes:
                size_counts[size] = size_counts.get(size, 0) + 1

            result["size_distribution"] = [
                {"size": size, "count": count}
                for size, count in sorted(size_counts.items())
            ]

            # Identify isolated nodes (components of size 1)
            isolated_nodes = [next(iter(comp)) for comp in components if len(comp) == 1]
            if isolated_nodes:
                result["isolated_nodes"] = isolated_nodes
                result["num_isolated_nodes"] = len(isolated_nodes)

        # Additional connectivity metrics
        if not graph.is_directed():
            # For undirected graphs, add articulation points and bridges
            if graph.number_of_nodes() > MIN_NODES_FOR_CONNECTIVITY:
                try:
                    articulation_points = list(nx.articulation_points(graph))
                    bridges = list(nx.bridges(graph))

                    result["articulation_points"] = articulation_points[
                        :20
                    ]  # Limit to 20
                    result["num_articulation_points"] = len(articulation_points)
                    result["bridges"] = bridges[:20]  # Limit to 20
                    result["num_bridges"] = len(bridges)
                except Exception as e:
                    logger.debug(f"Failed to compute critical nodes: {e}")

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("connected_components", elapsed_time)
        operation_counter.increment("connected_components")

        logger.info(f"Component analysis completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "connected_components", "Component analysis successful", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error finding components: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("ComponentError", str(e))


@mcp.tool()
async def find_all_paths(
    graph_id: str,
    source: Union[str, int],
    target: Union[str, int],
    max_length: Optional[int] = None,
    max_paths: int = MAX_PATHS_DEFAULT,
) -> dict[str, Any]:
    """
    Find all simple paths between two nodes with constraints.

    Args:
        graph_id: ID of the graph
        source: Source node
        target: Target node
        max_length: Maximum path length (None for no limit)
        max_paths: Maximum number of paths to return (default 100)

    Returns:
        Path information including:
        - paths: List of all paths found
        - path_count: Total number of paths
        - length_distribution: Distribution of path lengths
        - shortest_path_length: Length of shortest path
        - longest_path_length: Length of longest path
    """
    start_time = time.time()
    logger.info(
        f"Finding all paths from '{source}' to '{target}' in graph '{graph_id}'"
    )

    try:
        graph = graph_manager.get_graph(graph_id)

        # Validate nodes
        if source not in graph:
            return GraphFormatter.format_error(
                "NodeNotFoundError", f"Source node '{source}' not found"
            )
        if target not in graph:
            return GraphFormatter.format_error(
                "NodeNotFoundError", f"Target node '{target}' not found"
            )

        # Check if path exists
        if not nx.has_path(graph, source, target):
            return GraphFormatter.format_error(
                "NoPathError", f"No path exists from '{source}' to '{target}'"
            )

        result = {}

        # Find all simple paths
        paths = []
        path_lengths = []

        try:
            path_generator = nx.all_simple_paths(
                graph, source, target, cutoff=max_length
            )

            for i, path in enumerate(path_generator):
                if i >= max_paths:
                    result["truncated"] = True
                    result["truncation_message"] = (
                        f"Results limited to {max_paths} paths"
                    )
                    break

                paths.append(path)
                path_lengths.append(len(path) - 1)

            result["paths"] = paths
            result["path_count"] = len(paths)

        except nx.NetworkXError as e:
            return GraphFormatter.format_error("PathError", str(e))

        # Analyze path lengths
        if path_lengths:
            # Length distribution
            length_counts = {}
            for length in path_lengths:
                length_counts[length] = length_counts.get(length, 0) + 1

            result["length_distribution"] = [
                {
                    "length": length,
                    "count": c,
                    "percentage": round(c / len(paths) * 100, DECIMAL_PLACES),
                }
                for length, c in sorted(length_counts.items())
            ]

            result["shortest_path_length"] = min(path_lengths)
            result["longest_path_length"] = max(path_lengths)
            result["average_path_length"] = round(
                sum(path_lengths) / len(path_lengths), DECIMAL_PLACES
            )

        # Add graph context
        result["graph_info"] = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "is_directed": graph.is_directed(),
        }

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("find_all_paths", elapsed_time)
        operation_counter.increment("find_all_paths")

        logger.info(f"Found {len(paths)} paths in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "find_all_paths", "Paths found successfully", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error finding paths: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("PathFindingError", str(e))


@mcp.tool()
async def path_analysis(
    graph_id: str, sample_size: Optional[int] = 1000
) -> dict[str, Any]:
    """
    Analyze path properties of a graph including diameter, radius, and eccentricity.

    Args:
        graph_id: ID of the graph
        sample_size: Number of node pairs to sample for large graphs

    Returns:
        Path analysis including:
        - average_shortest_path_length: Average distance between all node pairs
        - diameter: Maximum eccentricity
        - radius: Minimum eccentricity
        - eccentricity: Dict of node eccentricities
        - center: Nodes with eccentricity equal to radius
        - periphery: Nodes with eccentricity equal to diameter
        - path_length_distribution: Distribution of shortest path lengths
    """
    start_time = time.time()
    logger.info(f"Analyzing path properties for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)
        result = {}

        # Check connectivity
        if graph.is_directed():
            is_connected = nx.is_strongly_connected(graph)
            connectivity_type = "strongly_connected"
        else:
            is_connected = nx.is_connected(graph)
            connectivity_type = "connected"

        result["is_" + connectivity_type] = is_connected

        if not is_connected:
            # Analyze largest component instead
            if graph.is_directed():
                components = list(nx.strongly_connected_components(graph))
            else:
                components = list(nx.connected_components(graph))

            largest_component = max(components, key=len)
            subgraph = graph.subgraph(largest_component).copy()

            result["analyzing_largest_component"] = True
            result["largest_component_size"] = len(largest_component)
            result["component_fraction"] = (
                len(largest_component) / graph.number_of_nodes()
            )

            logger.info(
                f"Graph not fully connected, analyzing largest component ({len(largest_component)} nodes)"
            )
        else:
            subgraph = graph

        # Calculate eccentricity for all nodes
        try:
            eccentricity = nx.eccentricity(subgraph)
            result["eccentricity"] = eccentricity

            # Diameter and radius
            result["diameter"] = nx.diameter(subgraph)
            result["radius"] = nx.radius(subgraph)

            # Center and periphery
            result["center"] = nx.center(subgraph)
            result["periphery"] = nx.periphery(subgraph)

            # Eccentricity statistics
            ecc_values = list(eccentricity.values())
            result["eccentricity_stats"] = {
                "mean": round(np.mean(ecc_values), DECIMAL_PLACES),
                "std": round(np.std(ecc_values), DECIMAL_PLACES),
                "min": min(ecc_values),
                "max": max(ecc_values),
            }

        except Exception as e:
            logger.warning(f"Could not compute eccentricity: {e}")
            result["eccentricity_error"] = str(e)

        # Average shortest path length
        if subgraph.number_of_nodes() < 1000 or not sample_size:
            # Exact calculation for small graphs
            try:
                result["average_shortest_path_length"] = round(
                    nx.average_shortest_path_length(subgraph), 4
                )
                result["path_calculation_method"] = "exact"
            except Exception as e:
                logger.warning(f"Could not compute average path length: {e}")
                result["average_path_error"] = str(e)
        else:
            # Sample for large graphs
            logger.info(f"Large graph detected, sampling {sample_size} node pairs")
            nodes = list(subgraph.nodes())
            sampled_lengths = []

            for _ in range(sample_size):
                u, v = np.random.choice(nodes, 2, replace=False)
                try:
                    length = nx.shortest_path_length(subgraph, u, v)
                    sampled_lengths.append(length)
                except nx.NetworkXNoPath:
                    pass

            if sampled_lengths:
                result["average_shortest_path_length"] = round(
                    np.mean(sampled_lengths), 4
                )
                result["path_calculation_method"] = "sampled"
                result["sample_size"] = len(sampled_lengths)

        # Path length distribution (sample if large)
        if subgraph.number_of_nodes() < 100:
            # Full distribution for small graphs
            path_lengths = []
            for source in subgraph.nodes():
                lengths = nx.single_source_shortest_path_length(subgraph, source)
                path_lengths.extend(lengths.values())

            if path_lengths:
                hist, bins = np.histogram(path_lengths, bins=min(20, max(path_lengths)))
                result["path_length_distribution"] = {
                    "counts": hist.tolist(),
                    "bins": bins.tolist(),
                }

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("path_analysis", elapsed_time)
        operation_counter.increment("path_analysis")

        logger.info(f"Path analysis completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "path_analysis", "Analysis completed successfully", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error in path analysis: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("PathAnalysisError", str(e))


@mcp.tool()
async def cycle_detection(
    graph_id: str, max_cycle_length: Optional[int] = None, limit: int = 100
) -> dict[str, Any]:
    """
    Detect cycles in a graph with detailed analysis.

    Args:
        graph_id: ID of the graph
        max_cycle_length: Maximum length of cycles to find (None for no limit)
        limit: Maximum number of cycles to return

    Returns:
        Cycle analysis including:
        - has_cycles: Whether graph contains cycles
        - cycles: List of cycles found (limited by 'limit' parameter)
        - cycle_count: Number of cycles found
        - shortest_cycle: Shortest cycle found
        - cycle_basis: Minimum cycle basis (for undirected)
        - is_dag: Whether graph is a DAG (directed only)
    """
    start_time = time.time()
    logger.info(f"Detecting cycles in graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)
        result = {}

        if graph.is_directed():
            # Directed graph cycle detection
            result["is_dag"] = nx.is_directed_acyclic_graph(graph)
            result["has_cycles"] = not result["is_dag"]

            if result["has_cycles"]:
                # Find cycles
                cycles = []
                try:
                    cycle_gen = nx.simple_cycles(graph)

                    for i, cycle in enumerate(cycle_gen):
                        if i >= limit:
                            result["truncated"] = True
                            result["truncation_message"] = (
                                f"Results limited to {limit} cycles"
                            )
                            break

                        if max_cycle_length and len(cycle) > max_cycle_length:
                            continue

                        cycles.append(cycle)

                    result["cycles"] = cycles
                    result["cycle_count"] = len(cycles)

                    if cycles:
                        # Find shortest cycle
                        shortest = min(cycles, key=len)
                        result["shortest_cycle"] = {
                            "nodes": shortest,
                            "length": len(shortest),
                        }

                        # Cycle length distribution
                        lengths = [len(c) for c in cycles]
                        length_counts = {}
                        for length in lengths:
                            length_counts[length] = length_counts.get(length, 0) + 1

                        result["cycle_length_distribution"] = [
                            {"length": length, "count": c}
                            for length, c in sorted(length_counts.items())
                        ]

                except Exception as e:
                    logger.warning(f"Error finding simple cycles: {e}")
                    result["cycle_error"] = "Too many cycles to enumerate efficiently"

                # Find nodes involved in cycles
                try:
                    nodes_in_cycles = set()
                    for cycle in cycles[:1000]:  # Limit for performance
                        nodes_in_cycles.update(cycle)
                    result["nodes_in_cycles"] = len(nodes_in_cycles)
                    result["fraction_in_cycles"] = (
                        len(nodes_in_cycles) / graph.number_of_nodes()
                    )
                except Exception as e:
                    logger.debug(f"Failed to compute cycle metrics: {e}")

        else:
            # Undirected graph cycle detection
            try:
                cycle_basis = nx.cycle_basis(graph)
                result["has_cycles"] = len(cycle_basis) > 0
                result["cycle_basis"] = cycle_basis[:limit]
                result["num_independent_cycles"] = len(cycle_basis)

                if cycle_basis:
                    # Find shortest cycle in basis
                    shortest = min(cycle_basis, key=len)
                    result["shortest_cycle_in_basis"] = {
                        "nodes": shortest,
                        "length": len(shortest),
                    }

                    # Cycle length distribution
                    lengths = [len(c) for c in cycle_basis]
                    result["basis_length_stats"] = {
                        "mean": round(np.mean(lengths), DECIMAL_PLACES),
                        "min": min(lengths),
                        "max": max(lengths),
                    }

                # Check if graph is a tree
                result["is_tree"] = nx.is_tree(graph)
                result["is_forest"] = nx.is_forest(graph)

            except Exception as e:
                logger.error(f"Error finding cycle basis: {e}")
                result["cycle_error"] = str(e)

        # Find self-loops
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            result["self_loops"] = self_loops[:20]  # Limit output
            result["num_self_loops"] = len(self_loops)

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("cycle_detection", elapsed_time)
        operation_counter.increment("cycle_detection")

        logger.info(f"Cycle detection completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "cycle_detection", "Cycle analysis completed", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error in cycle detection: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("CycleDetectionError", str(e))


@mcp.tool()
async def flow_paths(
    graph_id: str,
    source: Union[str, int],
    target: Union[str, int],
    capacity: str = "capacity",
    flow_type: str = "maximum",
) -> dict[str, Any]:
    """
    Analyze flow paths in a directed graph including max flow and edge-disjoint paths.

    Args:
        graph_id: ID of the graph (must be directed)
        source: Source node
        target: Target (sink) node
        capacity: Edge attribute name for capacity (default 'capacity')
        flow_type: Type of flow analysis - 'maximum', 'edge_disjoint', 'node_disjoint'

    Returns:
        Flow analysis including:
        - flow_value: Maximum flow value
        - flow_dict: Flow on each edge
        - min_cut: Edges in minimum cut
        - min_cut_value: Value of minimum cut
        - disjoint_paths: Edge or node disjoint paths
        - path_capacities: Capacity of each path
    """
    start_time = time.time()
    logger.info(
        f"Analyzing flow paths from '{source}' to '{target}' in graph '{graph_id}'"
    )

    try:
        graph = graph_manager.get_graph(graph_id)

        # Validate directed graph
        if not graph.is_directed():
            return GraphFormatter.format_error(
                "GraphTypeError", "Flow analysis requires a directed graph"
            )

        # Validate nodes
        if source not in graph:
            return GraphFormatter.format_error(
                "NodeNotFoundError", f"Source node '{source}' not found"
            )
        if target not in graph:
            return GraphFormatter.format_error(
                "NodeNotFoundError", f"Target node '{target}' not found"
            )

        result = {}

        if flow_type in {"maximum", "all"}:
            # Maximum flow
            try:
                flow_value, flow_dict = nx.maximum_flow(
                    graph, source, target, capacity=capacity
                )
                result["maximum_flow"] = {
                    "flow_value": flow_value,
                    "flow_dict": dict(flow_dict),
                }

                # Minimum cut
                cut_value, (reachable, non_reachable) = nx.minimum_cut(
                    graph, source, target, capacity=capacity
                )
                min_cut_edges = [
                    (u, v) for u in reachable for v in graph[u] if v in non_reachable
                ]

                result["minimum_cut"] = {
                    "cut_value": cut_value,
                    "cut_edges": min_cut_edges,
                    "num_cut_edges": len(min_cut_edges),
                    "reachable_nodes": len(reachable),
                    "non_reachable_nodes": len(non_reachable),
                }

            except nx.NetworkXError as e:
                result["flow_error"] = str(e)

        if flow_type in {"edge_disjoint", "all"}:
            # Edge-disjoint paths
            try:
                edge_disjoint = list(nx.edge_disjoint_paths(graph, source, target))
                result["edge_disjoint_paths"] = {
                    "paths": edge_disjoint,
                    "num_paths": len(edge_disjoint),
                    "path_lengths": [len(p) - 1 for p in edge_disjoint],
                }

                # Calculate capacity of each path if edge capacities exist
                if any(capacity in data for _, _, data in graph.edges(data=True)):
                    path_capacities = []
                    for path in edge_disjoint:
                        min_capacity = float("inf")
                        for i in range(len(path) - 1):
                            edge_data = graph[path[i]][path[i + 1]]
                            if capacity in edge_data:
                                min_capacity = min(min_capacity, edge_data[capacity])
                        path_capacities.append(
                            min_capacity if min_capacity != float("inf") else None
                        )

                    result["edge_disjoint_paths"]["path_capacities"] = path_capacities

            except nx.NetworkXError as e:
                result["edge_disjoint_error"] = str(e)

        if flow_type in {"node_disjoint", "all"}:
            # Node-disjoint paths
            try:
                node_disjoint = list(nx.node_disjoint_paths(graph, source, target))
                result["node_disjoint_paths"] = {
                    "paths": node_disjoint,
                    "num_paths": len(node_disjoint),
                    "path_lengths": [len(p) - 1 for p in node_disjoint],
                }
            except nx.NetworkXError as e:
                result["node_disjoint_error"] = str(e)

        # Analyze connectivity
        try:
            edge_connectivity = nx.edge_connectivity(graph, source, target)
            node_connectivity = nx.node_connectivity(graph, source, target)

            result["connectivity"] = {
                "edge_connectivity": edge_connectivity,
                "node_connectivity": node_connectivity,
            }
        except Exception as e:
            logger.debug(f"Failed to compute connectivity metrics: {e}")

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("flow_paths", elapsed_time)
        operation_counter.increment("flow_paths")

        logger.info(f"Flow analysis completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "flow_paths", "Flow analysis completed", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error in flow analysis: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("FlowAnalysisError", str(e))


@mcp.tool()
async def clear_graph(graph_id: str) -> dict[str, Any]:
    """
    Clear all nodes and edges from a graph while preserving the graph instance.

    Args:
        graph_id: ID of the graph to clear

    Returns:
        Clear operation status
    """
    start_time = time.time()
    logger.info(f"Clearing graph '{graph_id}'")

    try:
        result = graph_manager.clear_graph(graph_id)

        elapsed_time = time.time() - start_time
        performance_monitor.record_operation("clear_graph", elapsed_time)
        operation_counter.increment("clear_graph")

        logger.info(f"Cleared graph '{graph_id}' in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "clear_graph", "Graph cleared successfully", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error clearing graph '{graph_id}': {e!s}")
        return GraphFormatter.format_error("ClearGraphError", str(e))


@mcp.tool()
async def subgraph_extraction(
    graph_id: str,
    method: str = "nodes",
    nodes: Optional[list[Union[str, int]]] = None,
    edges: Optional[list[tuple[Union[str, int], Union[str, int]]]] = None,
    k_hop: Optional[int] = None,
    center_node: Optional[Union[str, int]] = None,
    condition: Optional[str] = None,
    create_new: bool = True,
    new_graph_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Extract subgraphs based on various criteria.

    Args:
        graph_id: ID of the source graph
        method: Extraction method - 'nodes', 'edges', 'k_hop', 'largest_component', 'condition'
        nodes: Node list for 'nodes' method
        edges: Edge list for 'edges' method
        k_hop: Number of hops for 'k_hop' method
        center_node: Center node for 'k_hop' method
        condition: Node/edge attribute condition (e.g., "weight > 0.5")
        create_new: Whether to create a new graph or return subgraph data
        new_graph_id: ID for the new graph (auto-generated if not provided)

    Returns:
        Subgraph information including:
        - subgraph_id: ID of created subgraph (if create_new=True)
        - num_nodes: Number of nodes in subgraph
        - num_edges: Number of edges in subgraph
        - nodes: List of nodes in subgraph
        - extraction_stats: Statistics about the extraction
    """
    start_time = time.time()
    logger.info(f"Extracting subgraph from '{graph_id}' using method '{method}'")

    try:
        graph = graph_manager.get_graph(graph_id)
        result = {}
        subgraph = None

        if method == "nodes":
            # Extract by node list
            if not nodes:
                return GraphFormatter.format_error(
                    "ValidationError",
                    "Node list required for 'nodes' extraction method",
                )

            # Validate nodes exist
            missing_nodes = [n for n in nodes if n not in graph]
            if missing_nodes:
                logger.warning(f"Nodes not found in graph: {missing_nodes}")

            valid_nodes = [n for n in nodes if n in graph]
            subgraph = graph.subgraph(valid_nodes).copy()

            result["extraction_info"] = {
                "requested_nodes": len(nodes),
                "valid_nodes": len(valid_nodes),
                "missing_nodes": missing_nodes[:10],  # Limit output
            }

        elif method == "edges":
            # Extract by edge list
            if not edges:
                return GraphFormatter.format_error(
                    "ValidationError",
                    "Edge list required for 'edges' extraction method",
                )

            # Create subgraph from edges
            subgraph = graph.edge_subgraph(edges).copy()

            result["extraction_info"] = {
                "requested_edges": len(edges),
                "valid_edges": subgraph.number_of_edges(),
            }

        elif method == "k_hop":
            # Extract k-hop neighborhood
            if center_node is None:
                return GraphFormatter.format_error(
                    "ValidationError",
                    "Center node required for 'k_hop' extraction method",
                )

            if center_node not in graph:
                return GraphFormatter.format_error(
                    "NodeNotFoundError",
                    f"Center node '{center_node}' not found in graph",
                )

            if k_hop is None:
                k_hop = 1

            # Find k-hop neighbors
            neighbors = {center_node}
            for _ in range(k_hop):
                new_neighbors = set()
                for node in neighbors:
                    new_neighbors.update(graph.neighbors(node))
                neighbors.update(new_neighbors)

            subgraph = graph.subgraph(neighbors).copy()

            result["extraction_info"] = {
                "center_node": center_node,
                "k_hop": k_hop,
                "neighborhood_size": len(neighbors),
            }

        elif method == "largest_component":
            # Extract largest connected component
            if graph.is_directed():
                components = list(nx.weakly_connected_components(graph))
            else:
                components = list(nx.connected_components(graph))

            if not components:
                return GraphFormatter.format_error(
                    "EmptyGraphError", "No components found in graph"
                )

            largest = max(components, key=len)
            subgraph = graph.subgraph(largest).copy()

            result["extraction_info"] = {
                "total_components": len(components),
                "largest_component_size": len(largest),
                "fraction_of_graph": len(largest) / graph.number_of_nodes(),
            }

        elif method == "condition":
            # Extract by condition on node/edge attributes
            if not condition:
                return GraphFormatter.format_error(
                    "ValidationError",
                    "Condition required for 'condition' extraction method",
                )

            # Parse condition (simple implementation)
            # Format: "attribute operator value" e.g., "weight > 0.5"
            try:
                parts = condition.split()
                if len(parts) != 3:
                    msg = "Condition must be in format: 'attribute operator value'"
                    raise ValueError(msg)

                attr, op, value = parts

                # Convert value to appropriate type
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

                # Apply condition to nodes
                valid_nodes = []
                for node, data in graph.nodes(data=True):
                    if attr in data:
                        node_value = data[attr]
                        if op == "=" and node_value == value:
                            valid_nodes.append(node)
                        elif op == ">" and node_value > value:
                            valid_nodes.append(node)
                        elif op == "<" and node_value < value:
                            valid_nodes.append(node)
                        elif op == ">=" and node_value >= value:
                            valid_nodes.append(node)
                        elif op == "<=" and node_value <= value:
                            valid_nodes.append(node)
                        elif op == "!=" and node_value != value:
                            valid_nodes.append(node)

                if valid_nodes:
                    subgraph = graph.subgraph(valid_nodes).copy()
                else:
                    # Try edge condition
                    valid_edges = []
                    for u, v, data in graph.edges(data=True):
                        if attr in data:
                            edge_value = data[attr]
                            if op == "=" and edge_value == value:
                                valid_edges.append((u, v))
                            elif op == ">" and edge_value > value:
                                valid_edges.append((u, v))
                            elif op == "<" and edge_value < value:
                                valid_edges.append((u, v))
                            elif op == ">=" and edge_value >= value:
                                valid_edges.append((u, v))
                            elif op == "<=" and edge_value <= value:
                                valid_edges.append((u, v))
                            elif op == "!=" and edge_value != value:
                                valid_edges.append((u, v))

                    if valid_edges:
                        subgraph = graph.edge_subgraph(valid_edges).copy()
                    else:
                        subgraph = graph.subgraph([]).copy()  # Empty subgraph

                result["extraction_info"] = {
                    "condition": condition,
                    "nodes_matching": (
                        len(valid_nodes) if "valid_nodes" in locals() else 0
                    ),
                    "edges_matching": (
                        len(valid_edges) if "valid_edges" in locals() else 0
                    ),
                }

            except Exception as e:
                return GraphFormatter.format_error(
                    "ConditionError", f"Error parsing condition: {e!s}"
                )

        else:
            return GraphFormatter.format_error(
                "ValidationError", f"Unknown extraction method: {method}"
            )

        # Process extracted subgraph
        if subgraph is not None:
            result["num_nodes"] = subgraph.number_of_nodes()
            result["num_edges"] = subgraph.number_of_edges()
            result["density"] = (
                nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0
            )

            # Node list (limit for large subgraphs)
            if subgraph.number_of_nodes() <= 1000:
                result["nodes"] = list(subgraph.nodes())
            else:
                result["nodes"] = list(subgraph.nodes())[:100]
                result["nodes_truncated"] = True

            # Create new graph if requested
            if create_new:
                if not new_graph_id:
                    new_graph_id = f"{graph_id}_subgraph_{int(time.time())}"

                # Create new graph with same type as original
                graph_type = type(graph).__name__
                graph_manager.graphs[new_graph_id] = subgraph
                graph_manager.metadata[new_graph_id] = {
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "graph_type": graph_type,
                    "parent_graph": graph_id,
                    "extraction_method": method,
                }

                result["subgraph_id"] = new_graph_id
                result["created"] = True

        # Performance metrics
        elapsed_time = time.time() - start_time
        result["computation_time_ms"] = round(
            elapsed_time * MILLISECONDS_PER_SECOND, DECIMAL_PLACES
        )

        performance_monitor.record_operation("subgraph_extraction", elapsed_time)
        operation_counter.increment("subgraph_extraction")

        logger.info(f"Subgraph extraction completed in {elapsed_time:.3f}s")
        return GraphFormatter.format_success(
            "subgraph_extraction", "Extraction completed", result
        )

    except KeyError:
        error_msg = f"Graph '{graph_id}' not found"
        logger.error(error_msg)
        return GraphFormatter.format_error("GraphNotFoundError", error_msg)
    except Exception as e:
        logger.error(f"Error in subgraph extraction: {e!s}")
        logger.debug(traceback.format_exc())
        return GraphFormatter.format_error("SubgraphExtractionError", str(e))


# Phase 2: Advanced Analytics Tools


@mcp.tool()
async def advanced_community_detection(
    graph_id: str,
    algorithm: str = "auto",
    resolution: float = 1.0,
    seed: Optional[int] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Detect communities using advanced algorithms with auto-selection.

    Args:
        graph_id: ID of the graph
        algorithm: Algorithm - 'auto', 'louvain', 'girvan_newman', 'spectral',
                   'label_propagation', 'modularity_max'
        resolution: Resolution parameter (for resolution-based methods)
        seed: Random seed for reproducibility
        params: Additional algorithm-specific parameters

    Returns:
        Community structure with quality metrics
    """
    start_time = time.time()
    logger.info(f"Running advanced community detection on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        # Run community detection
        algorithm_params = params or {}
        result = CommunityDetection.detect_communities(
            graph,
            algorithm=algorithm,
            resolution=resolution,
            seed=seed,
            **algorithm_params,
        )

        performance_monitor.record_operation(
            "advanced_community_detection", time.time() - start_time
        )
        operation_counter.increment("advanced_community_detection")

        return GraphFormatter.format_success(
            "advanced_community_detection",
            f"Detected {result['num_communities']} communities",
            result,
        )

    except Exception as e:
        logger.error(f"Error in advanced community detection: {e!s}")
        return GraphFormatter.format_error("CommunityDetectionError", str(e))


@mcp.tool()
async def network_flow_analysis(
    graph_id: str,
    source: Union[str, int],
    sink: Union[str, int],
    capacity: str = "capacity",
    algorithm: str = "auto",
    flow_type: str = "max_flow",
) -> dict[str, Any]:
    """
    Analyze network flow with multiple algorithms.

    Args:
        graph_id: ID of the graph
        source: Source node
        sink: Sink node
        capacity: Edge attribute for capacities
        algorithm: Algorithm - 'auto', 'ford_fulkerson', 'edmonds_karp', 'dinic', 'preflow_push'
        flow_type: Analysis type - 'max_flow', 'min_cut', 'multi_commodity'

    Returns:
        Flow analysis results including flow value, flow dict, and cut sets
    """
    start_time = time.time()
    logger.info(f"Running network flow analysis on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        if flow_type == "max_flow":
            result = NetworkFlow.max_flow_analysis(
                graph, source, sink, capacity=capacity, algorithm=algorithm
            )
        elif flow_type == "min_cut":
            result = NetworkFlow.min_cut_analysis(
                graph, source, sink, capacity=capacity
            )
        else:
            return GraphFormatter.format_error(
                "InvalidFlowType", f"Unknown flow type: {flow_type}"
            )

        performance_monitor.record_operation("network_flow", time.time() - start_time)
        operation_counter.increment("network_flow")

        return GraphFormatter.format_success(
            "network_flow", "Flow analysis completed", result
        )

    except Exception as e:
        logger.error(f"Error in network flow analysis: {e!s}")
        return GraphFormatter.format_error("NetworkFlowError", str(e))


@mcp.tool()
async def generate_graph(
    graph_type: str,
    n: int,
    graph_id: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate synthetic graphs using various models.

    Args:
        graph_type: Type - 'random', 'scale_free', 'small_world', 'regular',
                    'tree', 'geometric', 'social'
        n: Number of nodes
        graph_id: ID for the generated graph (auto-generated if not provided)
        params: Model-specific parameters (e.g., m for BA model, p for ER model)

    Returns:
        Generated graph information and statistics
    """
    start_time = time.time()

    if not graph_id:
        graph_id = f"generated_{graph_type}_{int(time.time())}"

    logger.info(f"Generating {graph_type} graph with {n} nodes")

    try:
        # Generate the graph
        generator_params = params or {}
        if graph_type == "random":
            result = GraphGenerators.random_graph(n, **generator_params)
        elif graph_type == "scale_free":
            result = GraphGenerators.scale_free_graph(n, **generator_params)
        elif graph_type == "small_world":
            result = GraphGenerators.small_world_graph(n, **generator_params)
        elif graph_type == "regular":
            result = GraphGenerators.regular_graph(n, **generator_params)
        elif graph_type == "tree":
            result = GraphGenerators.tree_graph(n, **generator_params)
        elif graph_type == "geometric":
            result = GraphGenerators.geometric_graph(n, **generator_params)
        elif graph_type == "social":
            result = GraphGenerators.social_network_graph(n, **generator_params)
        else:
            return GraphFormatter.format_error(
                "InvalidGraphType", f"Unknown graph type: {graph_type}"
            )

        # Store the generated graph
        generated_graph = result["graph"]
        graph_manager.create_graph(graph_id, generated_graph)

        # Add graph_id to result
        result["graph_id"] = graph_id
        del result["graph"]  # Remove the graph object from response

        performance_monitor.record_operation("generate_graph", time.time() - start_time)
        operation_counter.increment("generate_graph")

        return GraphFormatter.format_success(
            "generate_graph", f"Generated {graph_type} graph '{graph_id}'", result
        )

    except Exception as e:
        logger.error(f"Error generating graph: {e!s}")
        return GraphFormatter.format_error("GraphGenerationError", str(e))


@mcp.tool()
async def bipartite_analysis(
    graph_id: str,
    analysis_type: str = "check",
    weight: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Analyze bipartite graphs with specialized algorithms.

    Args:
        graph_id: ID of the graph
        analysis_type: Type - 'check', 'projection', 'matching', 'clustering', 'communities'
        weight: Edge weight attribute (for weighted operations)
        params: Analysis-specific parameters

    Returns:
        Bipartite analysis results
    """
    start_time = time.time()
    logger.info(f"Running bipartite analysis on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        if analysis_type == "check":
            result = BipartiteAnalysis.is_bipartite(graph)
        elif analysis_type == "projection":
            nodes = params.get("nodes", None)
            result = BipartiteAnalysis.bipartite_projection(graph, nodes, weight=weight)
        elif analysis_type == "matching":
            result = BipartiteAnalysis.maximum_matching(graph, weight=weight)
        elif analysis_type == "clustering":
            analysis_params = params or {}
            result = BipartiteAnalysis.bipartite_clustering(graph, **analysis_params)
        elif analysis_type == "communities":
            analysis_params = params or {}
            result = BipartiteAnalysis.bipartite_communities(graph, **analysis_params)
        else:
            return GraphFormatter.format_error(
                "InvalidAnalysisType", f"Unknown analysis type: {analysis_type}"
            )

        performance_monitor.record_operation(
            "bipartite_analysis", time.time() - start_time
        )
        operation_counter.increment("bipartite_analysis")

        return GraphFormatter.format_success(
            "bipartite_analysis", "Analysis completed", result
        )

    except Exception as e:
        logger.error(f"Error in bipartite analysis: {e!s}")
        return GraphFormatter.format_error("BipartiteAnalysisError", str(e))


@mcp.tool()
async def directed_graph_analysis(
    graph_id: str,
    analysis_type: str = "dag_check",
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Analyze directed graphs with specialized algorithms.

    Args:
        graph_id: ID of the graph
        analysis_type: Type - 'dag_check', 'scc', 'topological_sort', 'tournament',
                      'bow_tie', 'hierarchy', 'temporal'
        params: Analysis-specific parameters

    Returns:
        Directed graph analysis results
    """
    start_time = time.time()
    logger.info(f"Running directed graph analysis on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        if not graph.is_directed():
            return GraphFormatter.format_error(
                "NotDirectedError", "Graph must be directed for this analysis"
            )

        if analysis_type == "dag_check":
            result = DirectedAnalysis.dag_analysis(graph)
        elif analysis_type == "scc":
            algorithm = params.get("algorithm", "tarjan")
            result = DirectedAnalysis.strongly_connected_components(
                graph, algorithm=algorithm
            )
        elif analysis_type == "topological_sort":
            result = (
                {"topological_order": list(nx.topological_sort(graph))}
                if nx.is_directed_acyclic_graph(graph)
                else {"error": "Graph contains cycles"}
            )
        elif analysis_type == "tournament":
            result = DirectedAnalysis.tournament_analysis(graph)
        elif analysis_type == "bow_tie":
            result = DirectedAnalysis.bow_tie_structure(graph)
        elif analysis_type == "hierarchy":
            result = DirectedAnalysis.hierarchy_metrics(graph)
        elif analysis_type == "temporal":
            analysis_params = params or {}
            result = DirectedAnalysis.temporal_analysis(graph, **analysis_params)
        else:
            return GraphFormatter.format_error(
                "InvalidAnalysisType", f"Unknown analysis type: {analysis_type}"
            )

        performance_monitor.record_operation(
            "directed_analysis", time.time() - start_time
        )
        operation_counter.increment("directed_analysis")

        return GraphFormatter.format_success(
            "directed_analysis", "Analysis completed", result
        )

    except Exception as e:
        logger.error(f"Error in directed graph analysis: {e!s}")
        return GraphFormatter.format_error("DirectedAnalysisError", str(e))


@mcp.tool()
async def specialized_algorithms(
    graph_id: str, algorithm: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Run specialized graph algorithms.

    Args:
        graph_id: ID of the graph
        algorithm: Algorithm - 'spanning_tree', 'coloring', 'max_clique',
                   'matching', 'vertex_cover', 'dominating_set', 'link_prediction'
        params: Algorithm-specific parameters

    Returns:
        Algorithm results
    """
    start_time = time.time()
    logger.info(f"Running {algorithm} on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        algorithm_params = params or {}
        if algorithm == "spanning_tree":
            result = SpecializedAlgorithms.spanning_trees(graph, **algorithm_params)
        elif algorithm == "coloring":
            result = SpecializedAlgorithms.graph_coloring(graph, **algorithm_params)
        elif algorithm == "max_clique":
            result = SpecializedAlgorithms.maximum_clique(graph, **algorithm_params)
        elif algorithm == "matching":
            result = SpecializedAlgorithms.graph_matching(graph, **algorithm_params)
        elif algorithm == "vertex_cover":
            result = SpecializedAlgorithms.vertex_cover(graph, **algorithm_params)
        elif algorithm == "dominating_set":
            result = SpecializedAlgorithms.dominating_set(graph, **algorithm_params)
        elif algorithm == "link_prediction":
            result = SpecializedAlgorithms.link_prediction(graph, **algorithm_params)
        else:
            return GraphFormatter.format_error(
                "InvalidAlgorithm", f"Unknown algorithm: {algorithm}"
            )

        performance_monitor.record_operation(
            f"specialized_{algorithm}", time.time() - start_time
        )
        operation_counter.increment(f"specialized_{algorithm}")

        return GraphFormatter.format_success(
            f"specialized_{algorithm}", "Algorithm completed", result
        )

    except Exception as e:
        logger.error(f"Error in {algorithm}: {e!s}")
        return GraphFormatter.format_error(f"{algorithm.title()}Error", str(e))


@mcp.tool()
async def ml_graph_analysis(
    graph_id: str, analysis_type: str, params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Machine learning-based graph analysis.

    Args:
        graph_id: ID of the graph
        analysis_type: Type - 'embeddings', 'features', 'similarity', 'anomaly'
        params: Analysis-specific parameters

    Returns:
        ML analysis results
    """
    start_time = time.time()
    logger.info(f"Running ML {analysis_type} on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        ml_params = params or {}
        if analysis_type == "embeddings":
            result = MLIntegration.node_embeddings(graph, **ml_params)
            # Limit embeddings in response for large graphs
            if "embeddings" in result and len(result["embeddings"]) > 100:
                sample_nodes = list(result["embeddings"].keys())[:10]
                result["sample_embeddings"] = {
                    node: result["embeddings"][node].tolist()[:10]
                    for node in sample_nodes
                }
                result["embeddings"] = (
                    "Full embeddings computed but not returned (too large)"
                )

        elif analysis_type == "features":
            result = MLIntegration.graph_features(graph, **ml_params)
        elif analysis_type == "similarity":
            graph2_id = ml_params.get("graph2_id")
            if not graph2_id:
                return GraphFormatter.format_error(
                    "MissingParameter", "graph2_id required for similarity analysis"
                )
            graph2 = graph_manager.get_graph(graph2_id)
            result = MLIntegration.similarity_metrics(graph, graph2, **ml_params)
        elif analysis_type == "anomaly":
            result = MLIntegration.anomaly_detection(graph, **ml_params)
        else:
            return GraphFormatter.format_error(
                "InvalidAnalysisType", f"Unknown analysis type: {analysis_type}"
            )

        performance_monitor.record_operation(
            f"ml_{analysis_type}", time.time() - start_time
        )
        operation_counter.increment(f"ml_{analysis_type}")

        return GraphFormatter.format_success(
            f"ml_{analysis_type}", "Analysis completed", result
        )

    except Exception as e:
        logger.error(f"Error in ML {analysis_type}: {e!s}")
        return GraphFormatter.format_error(f"ML{analysis_type.title()}Error", str(e))


@mcp.tool()
async def robustness_analysis(
    graph_id: str,
    analysis_type: str = "attack",
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Analyze network robustness and resilience.

    Args:
        graph_id: ID of the graph
        analysis_type: Type - 'attack', 'percolation', 'cascading', 'resilience'
        params: Analysis-specific parameters

    Returns:
        Robustness analysis results
    """
    start_time = time.time()
    logger.info(f"Running robustness {analysis_type} on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        robustness_params = params or {}
        if analysis_type == "attack":
            result = RobustnessAnalysis.attack_simulation(graph, **robustness_params)
        elif analysis_type == "percolation":
            result = RobustnessAnalysis.percolation_analysis(graph, **robustness_params)
        elif analysis_type == "cascading":
            initial_failures = robustness_params.get("initial_failures", [])
            if not initial_failures:
                # Default to random node
                initial_failures = [random.choice(list(graph.nodes()))]
            result = RobustnessAnalysis.cascading_failure(
                graph, initial_failures, **robustness_params
            )
        elif analysis_type == "resilience":
            result = RobustnessAnalysis.network_resilience(graph, **robustness_params)
        else:
            return GraphFormatter.format_error(
                "InvalidAnalysisType", f"Unknown analysis type: {analysis_type}"
            )

        performance_monitor.record_operation(
            f"robustness_{analysis_type}", time.time() - start_time
        )
        operation_counter.increment(f"robustness_{analysis_type}")

        return GraphFormatter.format_success(
            f"robustness_{analysis_type}", "Analysis completed", result
        )

    except Exception as e:
        logger.error(f"Error in robustness {analysis_type}: {e!s}")
        return GraphFormatter.format_error(
            f"Robustness{analysis_type.title()}Error", str(e)
        )


# Phase 3: Visualization & Integration Tools


@mcp.tool()
async def visualize_graph(
    graph_id: str,
    visualization_type: str = "static",
    layout: str = "spring",
    format: str = "png",
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create graph visualizations with various backends.

    Args:
        graph_id: ID of the graph
        visualization_type: Type - 'static', 'interactive', 'pyvis', 'specialized'
        layout: Layout algorithm - 'spring', 'circular', 'shell', 'kamada_kawai', 'hierarchical'
        format: Output format - 'png', 'svg', 'html', 'json'
        params: Additional visualization parameters

    Returns:
        Visualization data in requested format
    """
    start_time = time.time()
    logger.info(f"Creating {visualization_type} visualization for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        viz_params = params or {}
        if visualization_type == "static":
            # Matplotlib static visualization
            result = MatplotlibVisualizer.create_static_plot(
                graph, layout=layout, **viz_params
            )

        elif visualization_type == "interactive":
            # Plotly interactive visualization
            result = PlotlyVisualizer.create_interactive_plot(
                graph, layout=layout, **viz_params
            )

        elif visualization_type == "pyvis":
            # PyVis physics-based visualization
            result = PyvisVisualizer.create_interactive_network(
                graph,
                physics=viz_params.get("physics", "barnes_hut"),
                hierarchical=viz_params.get("hierarchical", False),
                **viz_params,
            )

        elif visualization_type == "specialized":
            # Specialized visualizations
            viz_subtype = viz_params.get("subtype", "heatmap")

            if viz_subtype == "heatmap":
                result = SpecializedVisualizations.heatmap_adjacency(
                    graph, **viz_params
                )
            elif viz_subtype == "chord":
                result = SpecializedVisualizations.chord_diagram(graph, **viz_params)
            elif viz_subtype == "sankey":
                if not graph.is_directed():
                    return GraphFormatter.format_error(
                        "InvalidGraphType", "Sankey diagram requires directed graph"
                    )
                result = SpecializedVisualizations.sankey_diagram(graph, **viz_params)
            elif viz_subtype == "dendrogram":
                result = SpecializedVisualizations.dendrogram_clustering(
                    graph, **viz_params
                )
            else:
                return GraphFormatter.format_error(
                    "InvalidVisualizationType",
                    f"Unknown specialized visualization: {viz_subtype}",
                )
        else:
            return GraphFormatter.format_error(
                "InvalidVisualizationType",
                f"Unknown visualization type: {visualization_type}",
            )

        performance_monitor.record_operation(
            "visualize_graph", time.time() - start_time
        )
        operation_counter.increment("visualize_graph")

        return GraphFormatter.format_success(
            "visualize_graph", "Visualization created", result
        )

    except Exception as e:
        logger.error(f"Error in visualization: {e!s}")
        return GraphFormatter.format_error("VisualizationError", str(e))


@mcp.tool()
async def visualize_3d(
    graph_id: str, layout: str = "spring3d", params: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Create 3D graph visualization.

    Args:
        graph_id: ID of the graph
        layout: 3D layout algorithm
        params: Additional parameters

    Returns:
        3D visualization data
    """
    start_time = time.time()
    logger.info(f"Creating 3D visualization for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)
        viz_3d_params = params or {}
        result = PlotlyVisualizer.create_3d_plot(graph, layout=layout, **viz_3d_params)

        performance_monitor.record_operation("visualize_3d", time.time() - start_time)
        operation_counter.increment("visualize_3d")

        return GraphFormatter.format_success(
            "visualize_3d", "3D visualization created", result
        )

    except Exception as e:
        logger.error(f"Error in 3D visualization: {e!s}")
        return GraphFormatter.format_error("Visualization3DError", str(e))


@mcp.tool()
async def import_from_source(
    source_type: str,
    source_config: dict[str, Any],
    graph_id: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Import graph data from various sources with intelligent parsing.

    Args:
        source_type: Type - 'csv', 'json', 'database', 'api', 'excel'
        source_config: Source-specific configuration
        graph_id: ID for the imported graph (auto-generated if not provided)
        params: Additional import parameters

    Returns:
        Import results with graph metadata
    """
    start_time = time.time()

    if not graph_id:
        graph_id = f"imported_{source_type}_{int(time.time())}"

    logger.info(f"Importing from {source_type} to graph '{graph_id}'")

    try:
        import_params = params or {}
        if source_type == "csv":
            result = DataPipelines.csv_pipeline(
                filepath=source_config["filepath"], **source_config, **import_params
            )

        elif source_type == "json":
            result = DataPipelines.json_pipeline(
                filepath=source_config["filepath"], **source_config, **import_params
            )

        elif source_type == "database":
            result = DataPipelines.database_pipeline(
                connection_string=source_config["connection_string"],
                query=source_config["query"],
                **source_config,
                **import_params,
            )

        elif source_type == "api":
            # Run async API pipeline
            result = await DataPipelines.api_pipeline(
                base_url=source_config["base_url"],
                endpoints=source_config["endpoints"],
                **source_config,
                **import_params,
            )

        elif source_type == "excel":
            result = DataPipelines.excel_pipeline(
                filepath=source_config["filepath"], **source_config, **import_params
            )

        else:
            return GraphFormatter.format_error(
                "InvalidSourceType", f"Unknown source type: {source_type}"
            )

        # Store the imported graph
        imported_graph = result["graph"]
        graph_manager.create_graph(graph_id, imported_graph)

        # Add graph_id to result
        result["graph_id"] = graph_id
        del result["graph"]  # Remove graph object from response

        performance_monitor.record_operation(
            "import_from_source", time.time() - start_time
        )
        operation_counter.increment("import_from_source")

        return GraphFormatter.format_success(
            "import_from_source",
            f"Imported {result['num_nodes']} nodes, {result['num_edges']} edges",
            result,
        )

    except Exception as e:
        logger.error(f"Error importing from {source_type}: {e!s}")
        return GraphFormatter.format_error("ImportError", str(e))


@mcp.tool()
async def batch_graph_analysis(
    graph_ids: list[str],
    operations: list[dict[str, Any]],
    parallel: bool = True,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Process multiple graphs with batch operations.

    Args:
        graph_ids: List of graph IDs to process
        operations: List of operations to perform
        parallel: Use parallel processing
        params: Additional parameters

    Returns:
        Batch analysis results
    """
    start_time = time.time()
    logger.info(f"Starting batch analysis for {len(graph_ids)} graphs")

    try:
        # Prepare graphs
        graphs = []
        for gid in graph_ids:
            try:
                graph = graph_manager.get_graph(gid)
                graphs.append((gid, graph))
            except Exception as e:
                logger.warning(f"Graph {gid} not found, skipping: {e}")

        # Run batch analysis
        batch_params = params or {}
        results = await enterprise_features.batch_analysis(
            graphs=graphs, operations=operations, parallel=parallel, **batch_params
        )

        performance_monitor.record_operation("batch_analysis", time.time() - start_time)
        operation_counter.increment("batch_analysis")

        return GraphFormatter.format_success(
            "batch_analysis", "Batch analysis completed", results
        )

    except Exception as e:
        logger.error(f"Error in batch analysis: {e!s}")
        return GraphFormatter.format_error("BatchAnalysisError", str(e))


@mcp.tool()
async def create_analysis_workflow(
    graph_id: str,
    workflow_steps: list[dict[str, Any]],
    cache_results: bool = True,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Execute analysis workflow with chained operations.

    Args:
        graph_id: ID of the graph
        workflow_steps: List of workflow step configurations
        cache_results: Cache intermediate results
        params: Additional parameters

    Returns:
        Workflow execution results
    """
    start_time = time.time()
    logger.info(f"Executing workflow on graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        results = enterprise_features.analysis_workflow(
            graph=graph,
            workflow_config=workflow_steps,
            cache_intermediate=cache_results,
        )

        performance_monitor.record_operation(
            "analysis_workflow", time.time() - start_time
        )
        operation_counter.increment("analysis_workflow")

        return GraphFormatter.format_success(
            "analysis_workflow", "Workflow completed", results
        )

    except Exception as e:
        logger.error(f"Error in analysis workflow: {e!s}")
        return GraphFormatter.format_error("WorkflowError", str(e))


@mcp.tool()
async def generate_report(
    analysis_data: dict[str, Any],
    report_format: str = "pdf",
    template: str = "default",
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Generate analysis report in various formats.

    Args:
        analysis_data: Analysis results to include
        report_format: Format - 'pdf', 'html'
        template: Report template
        params: Additional parameters

    Returns:
        Generated report data
    """
    start_time = time.time()
    logger.info(f"Generating {report_format} report")

    try:
        report_params = params or {}
        report_content = enterprise_features.report_generation(
            graph_analysis=analysis_data,
            template=template,
            format=report_format,
            **report_params,
        )

        if report_format == "pdf":
            # Encode PDF as base64
            report_b64 = base64.b64encode(report_content).decode()
            result = {
                "format": "pdf",
                "content_base64": report_b64,
                "size_bytes": len(report_content),
            }
        else:
            # HTML as string
            result = {
                "format": "html",
                "content": report_content,
                "size_bytes": len(report_content.encode()),
            }

        performance_monitor.record_operation(
            "generate_report", time.time() - start_time
        )
        operation_counter.increment("generate_report")

        return GraphFormatter.format_success(
            "generate_report", "Report generated", result
        )

    except Exception as e:
        logger.error(f"Error generating report: {e!s}")
        return GraphFormatter.format_error("ReportGenerationError", str(e))


@mcp.tool()
async def setup_monitoring(
    graph_id: str,
    alert_rules: list[dict[str, Any]],
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Setup monitoring and alerts for a graph.

    Args:
        graph_id: ID of the graph to monitor
        alert_rules: List of alert rule configurations
        params: Additional parameters

    Returns:
        Monitoring setup confirmation
    """
    start_time = time.time()
    logger.info(f"Setting up monitoring for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        # Check current state and trigger any immediate alerts
        monitoring_params = params or {}
        triggered_alerts = enterprise_features.alert_system(
            graph=graph, alert_rules=alert_rules, **monitoring_params
        )

        # Store alert configuration for future use
        monitoring_config = {
            "graph_id": graph_id,
            "alert_rules": alert_rules,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "initial_alerts": triggered_alerts,
        }

        performance_monitor.record_operation(
            "setup_monitoring", time.time() - start_time
        )
        operation_counter.increment("setup_monitoring")

        return GraphFormatter.format_success(
            "setup_monitoring",
            f"Monitoring configured with {len(alert_rules)} rules",
            monitoring_config,
        )

    except Exception as e:
        logger.error(f"Error setting up monitoring: {e!s}")
        return GraphFormatter.format_error("MonitoringSetupError", str(e))


@mcp.tool()
async def create_dashboard(
    graph_id: str,
    visualizations: Optional[list[str]] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create interactive dashboard with multiple visualizations.

    Args:
        graph_id: ID of the graph
        visualizations: List of visualization types to include
        params: Additional parameters

    Returns:
        Dashboard data
    """
    start_time = time.time()
    logger.info(f"Creating dashboard for graph '{graph_id}'")

    try:
        graph = graph_manager.get_graph(graph_id)

        dashboard_params = params or {}
        result = SpecializedVisualizations.create_dashboard(
            graph=graph, visualizations=visualizations, **dashboard_params
        )

        performance_monitor.record_operation(
            "create_dashboard", time.time() - start_time
        )
        operation_counter.increment("create_dashboard")

        return GraphFormatter.format_success(
            "create_dashboard", "Dashboard created", result
        )

    except Exception as e:
        logger.error(f"Error creating dashboard: {e!s}")
        return GraphFormatter.format_error("DashboardError", str(e))


@mcp.resource("graph://{graph_id}")
async def get_graph_resource(graph_id: str) -> list[TextContent]:
    """
    Provide graph data as an MCP resource.

    Args:
        graph_id: ID of the graph

    Returns:
        Graph data as text content
    """
    try:
        graph = graph_manager.get_graph(graph_id)
        data = GraphIOHandler.export_graph(graph, "json")

        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except Exception as e:
        return [
            TextContent(type="text", text=f"Error retrieving graph {graph_id}: {e!s}")
        ]


@mcp.resource("graphs://list")
async def list_graphs_resource() -> list[TextContent]:
    """
    List all available graphs as an MCP resource.

    Returns:
        List of graphs as text content
    """
    try:
        graphs = graph_manager.list_graphs()

        content = "Available Graphs:\n\n"
        for graph in graphs:
            content += f"- {graph['graph_id']} ({graph['graph_type']})\n"
            content += f"  Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}\n"
            content += f"  Created: {graph['metadata']['created_at']}\n\n"

        return [TextContent(type="text", text=content)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing graphs: {e!s}")]


def main() -> None:
    """Run the NetworkX MCP server."""
    # Import datetime for the import_graph tool
    globals()["datetime"] = datetime

    # Get transport method from environment or default
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    # Get port from environment or command line (for SSE transport)
    port = 8765  # Changed default port to avoid conflicts

    # Get host from environment or default to localhost for security
    host = os.environ.get("MCP_HOST", "127.0.0.1")

    # Check for port in environment
    if os.environ.get("MCP_PORT"):
        port = int(os.environ.get("MCP_PORT"))

    # Check for transport and port in command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["stdio", "sse", "streamable-http"]:
            transport = sys.argv[1]
            if len(sys.argv) > 2:
                try:
                    port = int(sys.argv[2])
                except ValueError:
                    logger.warning(
                        f"Invalid port '{sys.argv[2]}', using default port {port}"
                    )
        else:
            # Assume it's a port number
            try:
                port = int(sys.argv[1])
                transport = "sse"  # Default to SSE when port is specified
            except ValueError:
                logger.warning(f"Invalid argument '{sys.argv[1]}', using defaults")

    logger.info("Starting NetworkX MCP Server")
    logger.info(f"Transport: {transport}")

    if transport in ["sse", "streamable-http"]:
        logger.info(f"Port: {port}")
        logger.info(f"Host: {host}")
        if host == "0.0.0.0":
            logger.warning(
                "SECURITY WARNING: Server is binding to all interfaces (0.0.0.0)"
            )
            logger.warning(
                "This exposes the server to external connections. Use 127.0.0.1 for localhost only."
            )
        logger.info(f"Server will be available at http://{host}:{port}")
    else:
        logger.info("Using stdio transport - communicate via standard input/output")

    logger.info("Press Ctrl+C to stop the server")

    try:
        # Run the server using FastMCP's built-in run method
        if transport == "stdio":
            mcp.run()
        elif transport == "sse":
            # WARNING: Set MCP_HOST=0.0.0.0 to bind to all interfaces (security risk)
            mcp.run(transport="sse", port=port, host=host)
        elif transport == "streamable-http":
            # WARNING: Set MCP_HOST=0.0.0.0 to bind to all interfaces (security risk)
            mcp.run(transport="streamable-http", port=port, host=host)
        else:
            logger.error(f"Unknown transport: {transport}")
            logger.info("Valid transports: stdio, sse, streamable-http")
            sys.exit(1)

    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use!")
            logger.info("Try one of these options:")
            logger.info(f"  1. Kill the process: lsof -ti:{port} | xargs kill -9")
            logger.info(
                "  2. Use a different port: python -m networkx_mcp.server sse 8766"
            )
            logger.info(
                "  3. Set environment variable: MCP_PORT=8766 python -m networkx_mcp.server"
            )
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")


if __name__ == "__main__":
    main()
