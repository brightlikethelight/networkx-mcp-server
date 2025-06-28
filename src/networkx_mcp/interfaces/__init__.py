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

from .base import (
    GraphAnalyzer,
    Visualizer, 
    Storage,
    SecurityValidator,
    BaseGraphTool,
    BaseAnalyzer,
    BaseVisualizer,
    ToolRegistry
)

from .plugin import Plugin, PluginManager

__all__ = [
    # Protocols
    "GraphAnalyzer",
    "Visualizer",
    "Storage", 
    "SecurityValidator",
    "ToolRegistry",
    
    # Base classes
    "BaseGraphTool",
    "BaseAnalyzer", 
    "BaseVisualizer",
    
    # Plugin system
    "Plugin",
    "PluginManager"
]

__version__ = "1.0.0"
