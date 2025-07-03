"""FastMCP compatibility layer for Pydantic v1/v2 conflicts."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class FastMCPCompat:
    """Compatibility wrapper for FastMCP that handles Pydantic version conflicts."""

    def __init__(self, name: str = "networkx-mcp", description: str | None = None):
        self.name = name
        self.description = description or "NetworkX MCP Server"
        self._tools: dict[str, Callable] = {}
        self._resources: dict[str, Any] = {}
        self._prompts: dict[str, Any] = {}
        self._server = None

        # Try to import the appropriate MCP implementation
        try:
            # First try FastMCP (requires Pydantic v2)
            from mcp.server.fastmcp import FastMCP

            self._mcp = FastMCP(name=name)
            self._use_fastmcp = True
            logger.info("Using FastMCP implementation")
        except ImportError as e:
            logger.warning(f"FastMCP not available: {e}")
            try:
                # Fall back to standard MCP server
                from mcp import Server

                self._mcp = Server(name=name)
                self._use_fastmcp = False
                logger.info("Using standard MCP Server implementation")
            except ImportError:
                # If even standard MCP fails, create a minimal implementation
                logger.warning(
                    "No MCP implementation available, using minimal compatibility mode"
                )
                self._mcp = None
                self._use_fastmcp = False

    def tool(self, description: str = "", name: str | None = None):
        """Decorator for registering tools."""

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self._tools[tool_name] = func

            if self._mcp and hasattr(self._mcp, "tool"):
                # Use the actual MCP tool decorator if available
                return self._mcp.tool(description=description, name=tool_name)(func)
            else:
                # Store for later registration
                func._tool_description = description
                func._tool_name = tool_name
                return func

        return decorator

    def resource(self, uri: str, description: str = "", name: str | None = None):
        """Decorator for registering resources."""

        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__
            self._resources[resource_name] = func

            if self._mcp and hasattr(self._mcp, "resource"):
                return self._mcp.resource(
                    uri=uri, description=description, name=resource_name
                )(func)
            else:
                func._resource_uri = uri
                func._resource_description = description
                func._resource_name = resource_name
                return func

        return decorator

    def prompt(
        self,
        name: str,
        description: str = "",
        arguments: list[dict[str, Any]] | None = None,
    ):
        """Decorator for registering prompts."""

        def decorator(func: Callable) -> Callable:
            self._prompts[name] = func

            if self._mcp and hasattr(self._mcp, "prompt"):
                return self._mcp.prompt(
                    name=name, description=description, arguments=arguments
                )(func)
            else:
                func._prompt_name = name
                func._prompt_description = description
                func._prompt_arguments = arguments
                return func

        return decorator

    def run(self, transport: str = "stdio"):
        """Run the MCP server."""
        if self._use_fastmcp and self._mcp:
            # FastMCP style
            self._mcp.run(transport=transport)
        elif self._mcp:
            # Standard MCP style - needs async handling
            import asyncio

            from mcp.server.stdio import stdio_server

            async def serve():
                async with stdio_server() as (read_stream, write_stream):
                    await self._mcp.run(read_stream, write_stream)

            try:
                asyncio.run(serve())
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
        else:
            # Minimal fallback implementation
            logger.error("No MCP implementation available to run server")
            self._run_minimal_server()

    def _run_minimal_server(self):
        """Minimal server implementation for testing."""
        logger.info(f"Starting minimal compatibility server: {self.name}")
        logger.info(f"Registered tools: {list(self._tools.keys())}")
        logger.info(f"Registered resources: {list(self._resources.keys())}")
        logger.info(f"Registered prompts: {list(self._prompts.keys())}")

        # Simple REPL for testing
        print(f"\n{self.name} - Minimal Compatibility Mode")
        print(f"Description: {self.description}")
        print("\nAvailable tools:")
        for tool_name in self._tools:
            print(f"  - {tool_name}")

        print("\nServer is running in minimal mode. Press Ctrl+C to exit.")

        try:
            while True:
                # Just keep the server "running"
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")

    # Proxy common methods to underlying implementation
    def __getattr__(self, name):
        """Proxy attribute access to underlying MCP implementation."""
        if self._mcp and hasattr(self._mcp, name):
            return getattr(self._mcp, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
