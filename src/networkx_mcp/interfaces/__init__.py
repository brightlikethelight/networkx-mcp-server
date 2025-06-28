"""Public interfaces for NetworkX MCP Server.

This package defines the public interfaces and protocols that all components
of the NetworkX MCP Server should implement. These interfaces enable:

1. Plugin development
2. Clean architecture
3. Testability
4. Extensibility

Example usage:

    from networkx_mcp.interfaces import GraphAnalyzer, BaseGraphTool

    class MyAnalyzer(BaseGraphTool):
        async def execute(self, graph, **params):
            return {"result": "analysis complete"}
"""

from networkx_mcp.interfaces.base import BaseAnalyzer
from networkx_mcp.interfaces.base import BaseGraphTool
from networkx_mcp.interfaces.base import BaseVisualizer
from networkx_mcp.interfaces.base import GraphAnalyzer
from networkx_mcp.interfaces.base import SecurityValidator
from networkx_mcp.interfaces.base import Storage
from networkx_mcp.interfaces.base import ToolRegistry
from networkx_mcp.interfaces.base import Visualizer
from networkx_mcp.interfaces.plugin import Plugin
from networkx_mcp.interfaces.plugin import PluginManager


__all__ = [
    "BaseAnalyzer",
    # Base classes
    "BaseGraphTool",
    "BaseVisualizer",
    # Protocols
    "GraphAnalyzer",
    # Plugin system
    "Plugin",
    "PluginManager",
    "SecurityValidator",
    "Storage",
    "ToolRegistry",
    "Visualizer"
]

__version__ = "1.0.0"
