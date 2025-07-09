"""Minimal working MCP server for NetworkX operations."""
import logging
from typing import Dict, List, Any, Optional

import networkx as nx
from mcp.server.fastmcp import FastMCP

# Import the WORKING graph manager and algorithms
from .core.graph_operations import GraphManager
from .core.algorithms import GraphAlgorithms

logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP("networkx-mcp")

# Initialize managers
graph_manager = GraphManager()
algorithms = GraphAlgorithms()


@mcp.tool()
def create_graph(graph_id: str, directed: bool = False) -> Dict[str, Any]:
    """Create a new graph.
    
    Args:
        graph_id: Unique identifier for the graph
        directed: Whether the graph is directed (default: False)
        
    Returns:
        Dictionary with creation status and metadata
    """
    graph_type = "DiGraph" if directed else "Graph"
    return graph_manager.create_graph(graph_id, graph_type=graph_type)


@mcp.tool()
def add_nodes(graph_id: str, nodes: List[str]) -> Dict[str, Any]:
    """Add nodes to a graph.
    
    Args:
        graph_id: Graph identifier
        nodes: List of node identifiers to add
        
    Returns:
        Dictionary with operation status
    """
    return graph_manager.add_nodes_from(graph_id, nodes)


@mcp.tool()
def add_edges(graph_id: str, edges: List[List[str]]) -> Dict[str, Any]:
    """Add edges to a graph.
    
    Args:
        graph_id: Graph identifier
        edges: List of edges as [source, target] pairs
        
    Returns:
        Dictionary with operation status
    """
    # Convert edge list to tuples
    edge_tuples = [(edge[0], edge[1]) for edge in edges]
    return graph_manager.add_edges_from(graph_id, edge_tuples)


@mcp.tool()
def get_graph_info(graph_id: str) -> Dict[str, Any]:
    """Get information about a graph.
    
    Args:
        graph_id: Graph identifier
        
    Returns:
        Dictionary with graph information (nodes, edges, type, etc.)
    """
    return graph_manager.get_graph_info(graph_id)


@mcp.tool()
def shortest_path(graph_id: str, source: str, target: str) -> Dict[str, Any]:
    """Find shortest path between two nodes.
    
    Args:
        graph_id: Graph identifier
        source: Source node
        target: Target node
        
    Returns:
        Dictionary with path or error message
    """
    try:
        graph = graph_manager.get_graph(graph_id)
        path = algorithms.shortest_path(graph, source, target)
        return {"success": True, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def centrality_measures(
    graph_id: str, 
    measures: List[str]
) -> Dict[str, Any]:
    """Calculate centrality measures for graph nodes.
    
    Args:
        graph_id: Graph identifier
        measures: List of centrality measures to calculate
                 (degree, betweenness, closeness, eigenvector)
        
    Returns:
        Dictionary with centrality results
    """
    try:
        graph = graph_manager.get_graph(graph_id)
        
        results = {}
        for measure in measures:
            if measure == "degree":
                results["degree"] = dict(nx.degree_centrality(graph))
            elif measure == "betweenness":
                results["betweenness"] = dict(nx.betweenness_centrality(graph))
            elif measure == "closeness":
                results["closeness"] = dict(nx.closeness_centrality(graph))
            elif measure == "eigenvector":
                try:
                    results["eigenvector"] = dict(nx.eigenvector_centrality(graph))
                except nx.NetworkXException:
                    results["eigenvector"] = "Could not compute (graph may not be connected)"
                    
        return {"success": True, "centrality": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def delete_graph(graph_id: str) -> Dict[str, Any]:
    """Delete a graph.
    
    Args:
        graph_id: Graph identifier
        
    Returns:
        Dictionary with deletion status
    """
    return graph_manager.delete_graph(graph_id)


def main():
    """Main entry point for MCP server."""
    import mcp.server.stdio
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the server
    asyncio.run(mcp.server.stdio.stdio_server(mcp.create_initialization_options()))


if __name__ == "__main__":
    main()