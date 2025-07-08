#!/usr/bin/env python3
"""Enhanced MCP Server with full protocol compliance."""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from functools import wraps
import json

from ..core.graph_operations import GraphManager
from ..core.algorithms import GraphAlgorithms
from ..security.input_validation import validate_id, validate_node_list, safe_error_message
from ..security.resource_limits import with_resource_limits

logger = logging.getLogger(__name__)


class MCPTool:
    """Represents an MCP tool with full metadata."""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        version: str = "1.0.0"
    ):
        self.name = name
        self.func = func
        self.description = description
        self.input_schema = input_schema or self._generate_input_schema(func)
        self.output_schema = output_schema or {"type": "object"}
        self.category = category or "general"
        self.version = version
        
    def _generate_input_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate JSON Schema from function signature."""
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            param_type = hints.get(param_name, Any)
            schema_type = self._python_type_to_json_schema(param_type)
            
            properties[param_name] = {
                "type": schema_type["type"],
                "description": f"Parameter {param_name}"
            }
            
            if "enum" in schema_type:
                properties[param_name]["enum"] = schema_type["enum"]
            
            if param.default == param.empty:
                required.append(param_name)
            else:
                properties[param_name]["default"] = param.default
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _python_type_to_json_schema(self, python_type) -> Dict[str, Any]:
        """Convert Python type to JSON Schema type."""
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        
        # Handle Optional types
        if hasattr(python_type, "__origin__"):
            if python_type.__origin__ is type(None):
                return {"type": "null"}
            
        # Handle literal types for enums
        if hasattr(python_type, "__args__"):
            # This would handle Literal types
            return {"type": "string", "enum": list(python_type.__args__)}
            
        return type_map.get(python_type, {"type": "string"})
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against schema."""
        # Simple validation - in production, use jsonschema library
        required = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})
        
        # Check required parameters
        for req in required:
            if req not in params:
                raise ValueError(f"Missing required parameter: {req}")
        
        # Check types (simplified)
        validated = {}
        for key, value in params.items():
            if key in properties:
                # Basic type checking
                expected_type = properties[key].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    validated[key] = str(value)
                elif expected_type == "integer" and not isinstance(value, int):
                    validated[key] = int(value)
                else:
                    validated[key] = value
            else:
                # Extra parameters - log warning but allow
                logger.warning(f"Unknown parameter {key} for tool {self.name}")
                validated[key] = value
                
        return validated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to MCP discovery format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "category": self.category,
            "version": self.version
        }


class EnhancedMCPServer:
    """MCP Server with full protocol compliance."""
    
    def __init__(self, name: str = "networkx-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, Callable] = {}
        self.prompts: Dict[str, Callable] = {}
        
        # Initialize components
        self.graph_manager = GraphManager()
        self.graph_algorithms = GraphAlgorithms()
        
        # Register built-in tools
        self._register_tools()
    
    def tool(
        self,
        description: str,
        name: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ):
        """Enhanced tool decorator with full metadata."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            
            # Create MCPTool instance
            tool = MCPTool(
                name=tool_name,
                func=func,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                category=category
            )
            
            self.tools[tool_name] = tool
            
            @wraps(func)
            def wrapper(**kwargs):
                try:
                    # Validate parameters
                    validated_params = tool.validate_params(kwargs)
                    
                    # Execute tool
                    result = func(**validated_params)
                    
                    # Ensure result matches output schema
                    if not isinstance(result, dict):
                        result = {"result": result}
                        
                    return result
                    
                except Exception as e:
                    logger.error(f"Tool {tool_name} error: {e}")
                    return {"error": safe_error_message(e)}
            
            return wrapper
        
        return decorator
    
    def _register_tools(self):
        """Register all graph tools with enhanced metadata."""
        
        # Graph Operations
        @self.tool(
            description="Create a new graph",
            category="graph_ops",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[a-zA-Z0-9_-]{1,100}$",
                        "description": "Unique graph identifier"
                    },
                    "graph_type": {
                        "type": "string",
                        "enum": ["undirected", "directed", "multi", "multi_directed"],
                        "default": "undirected",
                        "description": "Type of graph to create"
                    },
                    "data": {
                        "type": "object",
                        "properties": {
                            "nodes": {"type": "array", "items": {"type": ["string", "integer"]}},
                            "edges": {"type": "array", "items": {"type": "array"}}
                        },
                        "description": "Initial graph data"
                    }
                },
                "required": ["name"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "nodes": {"type": "integer"},
                    "edges": {"type": "integer"},
                    "error": {"type": "string"}
                }
            }
        )
        @with_resource_limits
        def create_graph(name: str, graph_type: str = "undirected", data: Optional[Dict] = None) -> Dict[str, Any]:
            """Create a new NetworkX graph."""
            try:
                safe_name = validate_id(name, "Graph name")
                
                # Map user-friendly types to NetworkX types
                type_map = {
                    "undirected": "Graph",
                    "directed": "DiGraph",
                    "multi": "MultiGraph",
                    "multi_directed": "MultiDiGraph",
                }
                
                nx_type = type_map.get(graph_type, "Graph")
                result = self.graph_manager.create_graph(safe_name, nx_type)
                
                # Add initial data if provided
                if data and result.get("created"):
                    if "nodes" in data:
                        self.graph_manager.add_nodes_from(safe_name, validate_node_list(data["nodes"]))
                    if "edges" in data:
                        self.graph_manager.add_edges_from(safe_name, data["edges"])
                
                graph = self.graph_manager.get_graph(safe_name)
                
                return {
                    "success": True,
                    "name": safe_name,
                    "type": graph_type,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Algorithm tools
        @self.tool(
            description="Find shortest path between two nodes",
            category="algorithms",
            input_schema={
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "description": "Graph identifier"},
                    "source": {"type": ["string", "integer"], "description": "Source node"},
                    "target": {"type": ["string", "integer"], "description": "Target node"},
                    "weight": {"type": "string", "description": "Edge attribute to use as weight"}
                },
                "required": ["graph_name", "source", "target"]
            }
        )
        @with_resource_limits
        def shortest_path(graph_name: str, source: Any, target: Any, weight: Optional[str] = None) -> Dict[str, Any]:
            """Find shortest path using Dijkstra or Bellman-Ford."""
            try:
                graph = self.graph_manager.get_graph(graph_name)
                result = self.graph_algorithms.shortest_path(graph, source, target, weight)
                return {
                    "success": True,
                    "path": result["path"],
                    "length": result["length"],
                    "weighted": weight is not None
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Add more tools...
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request with proper formatting."""
        # Validate JSON-RPC format
        if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
            return self._error_response("Invalid JSON-RPC version", -32600, request.get("id"))
        
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Route to appropriate handler
        if method == "tools/list":
            return self._handle_tool_discovery(request_id)
        elif method == "tools/call":
            return self._handle_tool_call(params, request_id)
        elif method == "initialize":
            return self._handle_initialize(params, request_id)
        else:
            return self._error_response(f"Unknown method: {method}", -32601, request_id)
    
    def _handle_tool_discovery(self, request_id: Any) -> Dict[str, Any]:
        """Handle tool discovery request."""
        tools_list = [tool.to_dict() for tool in self.tools.values()]
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": tools_list
            },
            "id": request_id
        }
    
    def _handle_tool_call(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle tool execution request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            return self._error_response("Missing tool name", -32602, request_id)
        
        if tool_name not in self.tools:
            return self._error_response(f"Unknown tool: {tool_name}", -32601, request_id)
        
        tool = self.tools[tool_name]
        
        try:
            # Execute tool
            result = tool.func(**arguments)
            
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return self._error_response(str(e), -32603, request_id)
    
    def _handle_initialize(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle initialization request with capabilities."""
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "1.0",
                "capabilities": {
                    "tools": {
                        "listChanged": True  # We support dynamic tool updates
                    },
                    "resources": {
                        "subscribe": False  # Not yet implemented
                    },
                    "prompts": {
                        "listChanged": True
                    }
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            },
            "id": request_id
        }
    
    def _error_response(self, message: str, code: int, request_id: Any) -> Dict[str, Any]:
        """Generate JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": request_id
        }
    
    async def run_stdio(self):
        """Run server with STDIO transport."""
        import sys
        import asyncio
        
        logger.info(f"Starting {self.name} MCP server v{self.version}")
        
        # Read from stdin, write to stdout
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        while True:
            try:
                # Read line from stdin
                line = await reader.readline()
                if not line:
                    break
                    
                # Parse JSON-RPC request
                request = json.loads(line.decode())
                
                # Handle request
                response = self.handle_request(request)
                
                # Send response
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"Server error: {e}")
                error_response = self._error_response(str(e), -32700, None)
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()


def create_enhanced_server() -> EnhancedMCPServer:
    """Create and configure enhanced MCP server."""
    server = EnhancedMCPServer()
    
    # Add resources
    from ..mcp.resources import GraphResources
    resources = GraphResources(server, server.graph_manager)
    
    # Add prompts
    from ..mcp.prompts import GraphPrompts
    prompts = GraphPrompts(server)
    
    return server


if __name__ == "__main__":
    import asyncio
    
    server = create_enhanced_server()
    asyncio.run(server.run_stdio())