"""Core NetworkX MCP functionality."""

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager

# DO NOT import GraphIOHandler here - it loads pandas (+35MB)!
# Use get_io_handler() for lazy loading when actually needed

__all__ = ["GraphAlgorithms", "GraphManager", "get_io_handler"]


def get_io_handler():
    """
    Lazy load IO handler only when needed.
    
    WARNING: This imports pandas/scipy and adds 50MB memory overhead!
    Only call when you actually need file I/O operations.
    """
    from networkx_mcp.core.io import GraphIOHandler
    return GraphIOHandler
