"""Mock MCP module for testing when MCP package is not available."""

from typing import Callable, Optional


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
        self._prompts = {}
        # Expose tools as a public attribute for compatibility
        self.tools = self._tools

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

    def prompt(self, name: Optional[str] = None):
        """Mock prompt decorator."""

        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            self._prompts[prompt_name] = func
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


class Resource:
    """Mock Resource class."""

    def __init__(self, uri: str, name: str, description: str = ""):
        self.uri = uri
        self.name = name
        self.description = description


class ResourceContent:
    """Mock ResourceContent class."""

    def __init__(self, uri: str, mimeType: str, text: str):
        self.uri = uri
        self.mimeType = mimeType
        self.text = text


class TextResourceContent(ResourceContent):
    """Mock TextResourceContent class."""

    pass


class Prompt:
    """Mock Prompt class."""

    def __init__(self, name: str, description: str, arguments: list = None):
        self.name = name
        self.description = description
        self.arguments = arguments or []


class PromptArgument:
    """Mock PromptArgument class."""

    def __init__(self, name: str, description: str, required: bool = False):
        self.name = name
        self.description = description
        self.required = required


class Types:
    """Mock types module."""

    TextContent = TextContent
    Resource = Resource
    ResourceContent = ResourceContent
    TextResourceContent = TextResourceContent
    Prompt = Prompt
    PromptArgument = PromptArgument


class MockMCP:
    """Mock MCP module."""

    Server = Server
    FastMCP = FastMCP
    server = type("server", (), {"models": ServerModels()})()
    types = Types()
