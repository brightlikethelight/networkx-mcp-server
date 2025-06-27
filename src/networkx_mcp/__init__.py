"""NetworkX MCP Server - A Model Context Protocol server for NetworkX graph operations."""

__version__ = "0.1.0"
__author__ = "NetworkX MCP Server Contributors"

from .core.algorithms import GraphAlgorithms
from .core.graph_operations import GraphManager
from .core.io_handlers import GraphIOHandler
from .server import mcp

__all__ = [
    "mcp",
    "GraphManager",
    "GraphAlgorithms",
    "GraphIOHandler",
]
