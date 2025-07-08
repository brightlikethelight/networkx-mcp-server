"""MCP protocol handler integrating JSON-RPC with NetworkX tools.

This module bridges the JSON-RPC protocol layer with the NetworkX MCP
server functionality, handling all MCP-specific methods.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..protocol import (
    JsonRpcHandler,
    MCPMessages,
    MCPErrorMessages,
    ServerInfo,
    ToolInfo,
    ResourceInfo,
    PromptInfo,
    PROTOCOL_VERSION
)

from ..mcp.tool_schemas import TOOL_SCHEMAS as SCHEMA_DEFINITIONS
from ..core.thread_safe_graph_manager import ThreadSafeGraphManager
from ..concurrency import ConnectionPool, RequestQueue, RequestPriority

logger = logging.getLogger(__name__)


class MCPProtocolHandler(JsonRpcHandler):
    """MCP protocol handler extending JSON-RPC handler."""
    
    def __init__(self):
        """Initialize MCP protocol handler."""
        super().__init__()
        self.server_info = ServerInfo(
            name="networkx-mcp-server",
            version="1.0.0"
        )
        self.client_info = None
        self.initialized = False
        
        # Initialize thread-safe components
        self.graph_manager = ThreadSafeGraphManager(max_graphs=1000)
        self.connection_pool = ConnectionPool(max_connections=50)
        self.request_queue = RequestQueue(max_queue_size=1000, max_workers=10)
        self._queue_started = False
        
        # Register MCP methods
        self._register_mcp_methods()
    
    async def _start_request_queue(self):
        """Start the request queue for processing."""
        await self.request_queue.start(self._process_tool_request)
        
    async def _process_tool_request(self, request):
        """Process a tool request in the queue."""
        # This is called by the request queue workers
        return request  # The actual processing happens in _handle_tool_call
    
    def _register_mcp_methods(self):
        """Register all MCP protocol methods."""
        # Core MCP methods
        self.register_method("initialize", self._handle_initialize)
        self.register_method("notifications/initialized", self._handle_initialized)
        self.register_method("notifications/cancelled", self._handle_cancelled)
        
        # Tools
        self.register_method("tools/list", self._handle_tools_list)
        self.register_method("tools/call", self._handle_tool_call)
        
        # Resources
        self.register_method("resources/list", self._handle_resources_list)
        self.register_method("resources/read", self._handle_resource_read)
        
        # Prompts
        self.register_method("prompts/list", self._handle_prompts_list)
        self.register_method("prompts/get", self._handle_prompt_get)
        
        # Logging
        self.register_method("logging/setLevel", self._handle_set_log_level)
        
        logger.info("MCP protocol methods registered")
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        # Parse request
        request_data = MCPMessages.parse_initialize_request(params)
        
        # Check protocol version compatibility
        client_version = request_data["protocolVersion"]
        if not self._is_protocol_compatible(client_version):
            raise ValueError(MCPErrorMessages.INVALID_PROTOCOL_VERSION)
        
        # Store client info
        self.client_info = request_data["clientInfo"]
        
        # Mark as initialized
        self.initialized = True
        
        logger.info(f"MCP initialized with client: {self.client_info.name} v{self.client_info.version}")
        
        # Return capabilities
        return MCPMessages.initialize_response(
            server_info=self.server_info,
            capabilities=MCPMessages.capabilities(),
            instructions="NetworkX MCP Server provides graph manipulation and analysis tools."
        )
    
    async def _handle_initialized(self, params: Any) -> None:
        """Handle initialized notification."""
        logger.info("Client confirmed initialization")
        # This is a notification, no response needed
        return None
    
    async def _handle_cancelled(self, params: Dict[str, Any]) -> None:
        """Handle cancelled notification."""
        request_id = params.get("requestId")
        reason = params.get("reason", "Unknown")
        logger.info(f"Request {request_id} cancelled: {reason}")
        # This is a notification, no response needed
        return None
    
    async def _handle_tools_list(self, params: Any) -> Dict[str, Any]:
        """Handle tools/list request."""
        if not self.initialized:
            raise ValueError("Not initialized")
        
        # Define tool metadata
        tool_definitions = {
            "create_graph": {
                "description": "Create a new graph with specified name and type",
                "inputSchema": SCHEMA_DEFINITIONS.get("create_graph", {"type": "object"})
            },
            "add_nodes": {
                "description": "Add nodes to an existing graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("add_nodes", {"type": "object"})
            },
            "add_edges": {
                "description": "Add edges to an existing graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("add_edges", {"type": "object"})
            },
            "graph_info": {
                "description": "Get information about a graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("graph_info", {"type": "object"})
            },
            "list_graphs": {
                "description": "List all available graphs",
                "inputSchema": {"type": "object", "properties": {}}
            },
            "delete_graph": {
                "description": "Delete a graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("delete_graph", {"type": "object"})
            },
            "shortest_path": {
                "description": "Find shortest path between two nodes",
                "inputSchema": SCHEMA_DEFINITIONS.get("shortest_path", {"type": "object"})
            },
            "node_degree": {
                "description": "Get degree of nodes in a graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("node_degree", {"type": "object"})
            },
            "connected_components": {
                "description": "Find connected components in a graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("connected_components", {"type": "object"})
            },
            "centrality_measures": {
                "description": "Calculate various centrality measures",
                "inputSchema": SCHEMA_DEFINITIONS.get("centrality_measures", {"type": "object"})
            },
            "clustering_coefficients": {
                "description": "Calculate clustering coefficients",
                "inputSchema": SCHEMA_DEFINITIONS.get("clustering_coefficient", {"type": "object"})
            },
            "minimum_spanning_tree": {
                "description": "Find minimum spanning tree",
                "inputSchema": SCHEMA_DEFINITIONS.get("minimum_spanning_tree", {"type": "object"})
            },
            "maximum_flow": {
                "description": "Calculate maximum flow between nodes",
                "inputSchema": SCHEMA_DEFINITIONS.get("maximum_flow", {"type": "object"})
            },
            "graph_coloring": {
                "description": "Color graph with minimum colors",
                "inputSchema": SCHEMA_DEFINITIONS.get("graph_coloring", {"type": "object"})
            },
            "community_detection": {
                "description": "Detect communities in graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("community_detection", {"type": "object"})
            },
            "cycles_detection": {
                "description": "Find cycles in graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("find_cycles", {"type": "object"})
            },
            "matching": {
                "description": "Find matching in graph",
                "inputSchema": SCHEMA_DEFINITIONS.get("matching", {"type": "object"})
            },
            "graph_statistics": {
                "description": "Get comprehensive graph statistics",
                "inputSchema": SCHEMA_DEFINITIONS.get("graph_statistics", {"type": "object"})
            },
            "manage_feature_flags": {
                "description": "Manage feature flags",
                "inputSchema": SCHEMA_DEFINITIONS.get("manage_feature_flags", {"type": "object"})
            },
            "resource_status": {
                "description": "Get server resource status",
                "inputSchema": {"type": "object", "properties": {}}
            }
        }
        
        # Convert to ToolInfo objects
        tools = []
        for tool_name, definition in tool_definitions.items():
            tools.append(ToolInfo(
                name=tool_name,
                description=definition["description"],
                inputSchema=definition["inputSchema"]
            ))
        
        return MCPMessages.tools_list_response(tools)
    
    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        if not self.initialized:
            raise ValueError("Not initialized")
        
        # Parse request
        request_data = MCPMessages.parse_tool_call_request(params)
        tool_name = request_data["name"]
        arguments = request_data["arguments"]
        
        try:
            # Use connection pool for rate limiting
            async with self.connection_pool.acquire_connection():
                # Map tool names to graph manager methods
                if tool_name == "create_graph":
                    result = await self.graph_manager.create_graph(
                        arguments.get("name"),
                        arguments.get("graph_type", "undirected")
                    )
                elif tool_name == "add_nodes":
                    result = await self.graph_manager.add_nodes(
                        arguments.get("graph_name"),
                        arguments.get("nodes", [])
                    )
                elif tool_name == "add_edges":
                    result = await self.graph_manager.add_edges(
                        arguments.get("graph_name"),
                        arguments.get("edges", [])
                    )
                elif tool_name == "graph_info":
                    result = await self.graph_manager.get_graph_info(
                        arguments.get("graph_name")
                    )
                elif tool_name == "list_graphs":
                    result = await self.graph_manager.list_graphs(
                        arguments.get("limit", 100),
                        arguments.get("offset", 0)
                    )
                elif tool_name == "delete_graph":
                    result = await self.graph_manager.delete_graph(
                        arguments.get("graph_name")
                    )
                elif tool_name == "shortest_path":
                    result = await self.graph_manager.get_shortest_path(
                        arguments.get("graph_name"),
                        arguments.get("source"),
                        arguments.get("target"),
                        arguments.get("weight")
                    )
                elif tool_name == "centrality_measures":
                    result = await self.graph_manager.centrality_measures(
                        arguments.get("graph_name"),
                        arguments.get("measures", ["degree"]),
                        arguments.get("normalized", True)
                    )
                elif tool_name == "resource_status":
                    # Get status from various components
                    result = {
                        "success": True,
                        "lock_stats": self.graph_manager.get_lock_stats(),
                        "connection_pool": self.connection_pool.get_stats(),
                        "request_queue": self.request_queue.get_queue_stats()
                    }
                else:
                    # Tool not implemented in thread-safe manager yet
                    return MCPMessages.tool_call_response(
                        content=MCPMessages.create_error_content(
                            f"Tool '{tool_name}' not yet implemented in thread-safe mode"
                        ),
                        isError=True
                    )
                
                # Check if result indicates an error
                if isinstance(result, dict) and not result.get("success", True):
                    return MCPMessages.tool_call_response(
                        content=MCPMessages.create_error_content(
                            result.get("error", "Operation failed")
                        ),
                        isError=True
                    )
                
                # Return success response
                return MCPMessages.tool_call_response(
                    content=MCPMessages.create_success_content(result)
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout acquiring connection for tool {tool_name}")
            return MCPMessages.tool_call_response(
                content=MCPMessages.create_error_content(
                    "Server overloaded, please try again"
                ),
                isError=True
            )
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return MCPMessages.tool_call_response(
                content=MCPMessages.create_error_content(str(e)),
                isError=True
            )
    
    async def _handle_resources_list(self, params: Any) -> Dict[str, Any]:
        """Handle resources/list request."""
        if not self.initialized:
            raise ValueError("Not initialized")
        
        # Currently no resources implemented
        resources = []
        
        return MCPMessages.resources_list_response(resources)
    
    async def _handle_resource_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request."""
        if not self.initialized:
            raise ValueError("Not initialized")
        
        request_data = MCPMessages.parse_resource_read_request(params)
        uri = request_data["uri"]
        
        # No resources implemented yet
        raise ValueError(f"{MCPErrorMessages.RESOURCE_NOT_FOUND}: {uri}")
    
    async def _handle_prompts_list(self, params: Any) -> Dict[str, Any]:
        """Handle prompts/list request."""
        if not self.initialized:
            raise ValueError("Not initialized")
        
        # Define some useful prompts
        prompts = [
            PromptInfo(
                name="analyze_graph",
                description="Analyze a graph and provide insights",
                arguments=[
                    {"name": "graph_name", "description": "Name of the graph to analyze", "required": True}
                ]
            ),
            PromptInfo(
                name="create_social_network",
                description="Create a social network graph with suggested structure",
                arguments=[
                    {"name": "num_users", "description": "Number of users in the network", "required": True},
                    {"name": "connectivity", "description": "Average connections per user", "required": False}
                ]
            )
        ]
        
        return MCPMessages.prompts_list_response(prompts)
    
    async def _handle_prompt_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt get request."""
        if not self.initialized:
            raise ValueError("Not initialized")
        
        request_data = MCPMessages.parse_prompt_get_request(params)
        prompt_name = request_data["name"]
        arguments = request_data["arguments"]
        
        # Simple prompt templates
        if prompt_name == "analyze_graph":
            graph_name = arguments.get("graph_name")
            return {
                "description": f"Analyze the graph '{graph_name}'",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Please analyze the graph '{graph_name}' and provide insights about its structure, connectivity, and important nodes."
                        }
                    }
                ]
            }
        elif prompt_name == "create_social_network":
            num_users = arguments.get("num_users", 10)
            connectivity = arguments.get("connectivity", 3)
            return {
                "description": f"Create a social network with {num_users} users",
                "messages": [
                    {
                        "role": "user", 
                        "content": {
                            "type": "text",
                            "text": f"Create a social network graph with {num_users} users where each user has approximately {connectivity} connections on average. Make it realistic with some users being more connected than others."
                        }
                    }
                ]
            }
        
        raise ValueError(f"{MCPErrorMessages.PROMPT_NOT_FOUND}: {prompt_name}")
    
    async def _handle_set_log_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logging level change request."""
        level = params.get("level", "info").upper()
        
        # Map to Python logging levels
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level in level_map:
            logging.getLogger().setLevel(level_map[level])
            logger.info(f"Log level set to {level}")
            return {}
        else:
            raise ValueError(f"Invalid log level: {level}")
    
    def _is_protocol_compatible(self, client_version: str) -> bool:
        """Check if client protocol version is compatible."""
        # For now, accept any version
        # In production, would check version compatibility
        return True
    
    async def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up MCP protocol handler")
        if self._queue_started:
            await self.request_queue.stop()
        await self.graph_manager.cleanup()