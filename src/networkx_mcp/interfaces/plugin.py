"""Plugin interface for extending NetworkX MCP Server."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from networkx_mcp.interfaces.base import BaseGraphTool


class Plugin(ABC):
    """Base class for NetworkX MCP Server plugins."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.tools = []

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin."""
        pass

    @abstractmethod
    def get_tools(self) -> List[BaseGraphTool]:
        """Get tools provided by this plugin."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "tools": [tool.name for tool in self.get_tools()]
        }

class PluginManager:
    """Manages plugins for the MCP server."""

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.tool_registry = {}

    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin."""
        if plugin.name in self.plugins:
            return False

        if not plugin.initialize():
            return False

        self.plugins[plugin.name] = plugin

        # Register plugin tools
        for tool in plugin.get_tools():
            self.tool_registry[tool.name] = tool

        return True

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin."""
        if name not in self.plugins:
            return False

        plugin = self.plugins[name]

        # Remove plugin tools
        for tool in plugin.get_tools():
            if tool.name in self.tool_registry:
                del self.tool_registry[tool.name]

        plugin.cleanup()
        del self.plugins[name]
        return True

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self.plugins.keys())

    def get_tool(self, name: str) -> Optional[BaseGraphTool]:
        """Get a tool from any plugin."""
        return self.tool_registry.get(name)
