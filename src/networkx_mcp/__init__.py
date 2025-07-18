"""NetworkX MCP Server - A Model Context Protocol server for NetworkX graphs."""

from networkx_mcp.__version__ import __version__


# Lazy imports to avoid import errors during build
def _get_graph_algorithms():
    """Lazy import of GraphAlgorithms to avoid build-time dependency issues."""
    try:
        from networkx_mcp.core.algorithms import GraphAlgorithms

        return GraphAlgorithms
    except ImportError:
        return None


def _get_graph_manager():
    """Lazy import of GraphManager to avoid build-time dependency issues."""
    try:
        from networkx_mcp.core.graph_operations import GraphManager

        return GraphManager
    except ImportError:
        return None


# Only expose version for build-time compatibility
__all__ = [
    "__version__",
]

# Conditionally add classes if dependencies are available
GraphAlgorithms = _get_graph_algorithms()
GraphManager = _get_graph_manager()

if GraphAlgorithms is not None:
    __all__.append("GraphAlgorithms")
if GraphManager is not None:
    __all__.append("GraphManager")
