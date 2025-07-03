"""
Algorithms handler for NetworkX MCP Server.

This module re-exports algorithm functions from the main server module.
"""

# Re-export functions from the main server module  
from ..server import (
    shortest_path,
    node_degree,
    graphs,
    mcp
)

# For backward compatibility
class AlgorithmsHandler:
    """Handler for graph algorithms."""
    
    def __init__(self):
        self.graphs = graphs
        self.mcp = mcp

# Create handler instance
algorithms_handler = AlgorithmsHandler()

# Export the functions for direct import
__all__ = [
    'shortest_path',
    'node_degree',
    'graphs',
    'mcp', 
    'AlgorithmsHandler',
    'algorithms_handler'
]
