"""Enhanced FastMCP compatibility layer with full MCP protocol support."""

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from typing import Any, Dict, List, Optional
from functools import wraps

from ..mcp.jsonrpc_handler import JsonRpcHandler, JsonRpcRequest, JsonRpcResponse
from ..mcp.tool_schemas import TOOL_SCHEMAS, OUTPUT_SCHEMAS

logger = logging.getLogger(__name__)


class MCPTool:
    """Enhanced tool wrapper with metadata and validation."""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.func = func
        self.description = description
        self.input_schema = input_schema or TOOL_SCHEMAS.get(name, {})
        self.output_schema = output_schema or OUTPUT_SCHEMAS.get(name, {"type": "object"})
        
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Basic parameter validation."""
        # In production, use jsonschema library
        if not self.input_schema:
            return params
            
        required = self.input_schema.get("required", [])
        for req in required:
            if req not in params:
                raise ValueError(f"Missing required parameter: {req}")
                
        return params


class EnhancedFastMCPCompat:
    """Enhanced compatibility wrapper with full MCP protocol support."""

    def __init__(self, name: str = "networkx-mcp", description: str | None = None, version: str = "1.0.0"):
        self.name = name
        self.description = description or "NetworkX MCP Server"
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self._server = None
        
        # JSON-RPC handler
        self.jsonrpc_handler = JsonRpcHandler(self)

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
                # If even standard MCP fails, use enhanced mode
                logger.info("Using enhanced MCP compatibility mode")
                self._mcp = None
                self._use_fastmcp = False

    def tool(
        self, 
        description: str = "", 
        name: str | None = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ):
        """Enhanced tool decorator with metadata and validation."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            
            # Create MCPTool wrapper
            tool = MCPTool(
                name=tool_name,
                func=func,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema
            )
            
            self.tools[tool_name] = tool

            # If underlying MCP exists, register with it
            if self._mcp and hasattr(self._mcp, "tool"):
                return self._mcp.tool(description=description, name=tool_name)(func)
            else:
                # Enhanced wrapper with validation
                @wraps(func)
                def wrapper(**kwargs):
                    try:
                        # Validate parameters
                        validated = tool.validate_params(kwargs)
                        # Execute
                        result = func(**validated)
                        # Ensure dict result
                        if not isinstance(result, dict):
                            result = {"result": result}
                        return result
                    except Exception as e:
                        logger.error(f"Tool {tool_name} error: {e}")
                        return {"error": str(e)}
                
                return wrapper

        return decorator

    def resource(self, uri: str, description: str = "", mime_type: str = "application/json"):
        """Enhanced resource decorator."""
        def decorator(func: Callable) -> Callable:
            self.resources[uri] = {
                "uri": uri,
                "description": description,
                "mimeType": mime_type,
                "handler": func
            }

            if self._mcp and hasattr(self._mcp, "resource"):
                try:
                    return self._mcp.resource(uri)(func)
                except Exception as e:
                    logger.debug(f"Native resource registration failed: {e}")
                    return func
            else:
                return func

        return decorator

    def prompt(
        self,
        name: str,
        description: str = "",
        arguments: list[dict[str, Any]] | None = None,
    ):
        """Enhanced prompt decorator."""
        def decorator(func: Callable) -> Callable:
            self.prompts[name] = {
                "name": name,
                "description": description,
                "arguments": arguments or [],
                "handler": func
            }

            if self._mcp and hasattr(self._mcp, "prompt"):
                try:
                    # FastMCP may not support all parameters
                    return self._mcp.prompt()(func)
                except Exception as e:
                    logger.debug(f"Native prompt registration failed: {e}")
                    return func
            else:
                return func

        return decorator
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema
            }
            for tool in self.tools.values()
        ]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all registered resources."""
        return [
            {
                "uri": res["uri"],
                "description": res.get("description", ""),
                "mimeType": res.get("mimeType", "application/json")
            }
            for res in self.resources.values()
        ]
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all registered prompts."""
        return [
            {
                "name": prompt["name"],
                "description": prompt.get("description", ""),
                "arguments": prompt.get("arguments", [])
            }
            for prompt in self.prompts.values()
        ]

    def run(self, transport: str = "stdio"):
        """Run the MCP server with enhanced protocol support."""
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
            # Enhanced implementation with full JSON-RPC support
            logger.info(f"Starting enhanced {self.name} server")
            asyncio.run(self._run_enhanced_server())

    async def _run_enhanced_server(self):
        """Run enhanced server with JSON-RPC protocol support."""
        logger.info(f"{self.name} MCP Server v{self.version}")
        logger.info(f"Tools: {len(self.tools)}, Resources: {len(self.resources)}, Prompts: {len(self.prompts)}")
        
        # Set up async stdio
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        try:
            await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        except Exception as e:
            logger.error(f"Failed to connect stdin: {e}")
            return
        
        # Main message loop
        while True:
            try:
                # Read line from stdin
                line_bytes = await reader.readline()
                if not line_bytes:
                    logger.info("EOF received, shutting down")
                    break
                
                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue
                
                logger.debug(f"Received: {line}")
                
                # Parse request
                request_obj = self.jsonrpc_handler.parse_message(line)
                
                if isinstance(request_obj, JsonRpcResponse):
                    # Parse error
                    response = request_obj
                else:
                    # Handle request
                    response = self.jsonrpc_handler.handle_request(request_obj)
                
                # Send response
                response_json = response.to_json()
                logger.debug(f"Sending: {response_json}")
                
                sys.stdout.write(response_json + "\n")
                sys.stdout.flush()
                
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                # Send error response
                error_response = JsonRpcResponse(
                    error={"code": -32603, "message": str(e)}
                ).to_json()
                sys.stdout.write(error_response + "\n")
                sys.stdout.flush()

    def handle_message(self, message: str) -> str:
        """Handle a single JSON-RPC message and return response."""
        request_obj = self.jsonrpc_handler.parse_message(message)
        
        if isinstance(request_obj, JsonRpcResponse):
            # Parse error
            return request_obj.to_json()
        else:
            # Handle request
            response = self.jsonrpc_handler.handle_request(request_obj)
            return response.to_json()

    # Proxy common methods to underlying implementation
    def __getattr__(self, name):
        """Proxy attribute access to underlying MCP implementation."""
        if self._mcp and hasattr(self._mcp, name):
            return getattr(self._mcp, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )