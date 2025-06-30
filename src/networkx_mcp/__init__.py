"""NetworkX MCP Server - A Model Context Protocol server for NetworkX graphs."""

__version__ = "0.1.0"
__author__ = "NetworkX MCP Server Contributors"

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.core.io_handlers import GraphIOHandler
from networkx_mcp.server import mcp

__all__ = [
    "GraphAlgorithms",
    "GraphIOHandler",
    "GraphManager",
    "mcp",
]
