"""NetworkX MCP Server - A Model Context Protocol server for NetworkX graphs."""

from networkx_mcp.__version__ import __version__
from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.core.io import GraphIOHandler
from networkx_mcp.server import NetworkXMCPServer

__all__ = [
    "__version__",
    "GraphAlgorithms",
    "GraphIOHandler",
    "GraphManager",
    "NetworkXMCPServer",
]

# Import handlers
try:
    pass
except ImportError:
    pass

# Import modular components
try:
    pass
except ImportError:
    pass

