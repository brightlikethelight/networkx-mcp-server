"""
Graph operations handler for NetworkX MCP Server.

This module re-exports graph operations from the main server module.
"""

# Re-export functions from the main server module
from ..server import (
    add_edges,
    add_nodes,
    create_graph,
    delete_graph,
    graphs,
    mcp,
)


# For backward compatibility
class GraphOpsHandler:
    """Handler for graph operations."""

    def __init__(self) -> None:
        self.graphs = graphs
        self.mcp = mcp


# Create handler instance
graph_ops_handler = GraphOpsHandler()

# Export the functions for direct import
__all__ = [
    "create_graph",
    "add_nodes",
    "add_edges",
    "delete_graph",
    "graphs",
    "mcp",
    "GraphOpsHandler",
    "graph_ops_handler",
]
