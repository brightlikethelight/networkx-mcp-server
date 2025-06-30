"""Mock MCP module for testing when MCP package is not available."""

from typing import Any, Dict, List, Optional, Callable


class TextContent:
    """Mock TextContent class."""
    
    def __init__(self, type: str, text: str):
        self.type = type
        self.text = text


class InitializationOptions:
    """Mock InitializationOptions class."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Server:
    """Mock MCP Server class with decorator support."""
    
    def __init__(self, name: str):
        self.name = name
        self.request_context = None
        self._tools = {}
        self._resources = {}
    
    def tool(self, name: Optional[str] = None):
        """Mock tool decorator."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self._tools[tool_name] = func
            return func
        return decorator
    
    def resource(self, pattern: str):
        """Mock resource decorator."""
        def decorator(func: Callable) -> Callable:
            self._resources[pattern] = func
            return func
        return decorator
    
    async def run(self, **kwargs):
        """Mock run method."""
        print(f"Mock MCP Server '{self.name}' running (MCP package not available)")
        print("Note: This is a stub implementation for testing purposes")
        # Keep the server "running" but do nothing
        import asyncio
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("Mock server stopped")


class FastMCP(Server):
    """Mock FastMCP class."""
    pass


# Mock module structure
class ServerModels:
    """Mock server.models module."""
    InitializationOptions = InitializationOptions


class Types:
    """Mock types module."""
    TextContent = TextContent


class MockMCP:
    """Mock MCP module."""
    Server = Server
    FastMCP = FastMCP
    server = type('server', (), {'models': ServerModels()})()
    types = Types()