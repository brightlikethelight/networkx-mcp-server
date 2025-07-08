#!/usr/bin/env python3
"""
NetworkX MCP Server - Minimal Implementation

This module provides a basic MCP (Model Context Protocol) server implementation
that exposes NetworkX graph operations through the MCP protocol over stdio transport.

It implements JSON-RPC 2.0 message framing and core MCP methods including:
- initialize/initialized for client handshake
- tools/list for tool discovery  
- tools/call for tool execution
- Basic NetworkX graph operations as MCP tools
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import networkx as nx

# Add parent directory to Python path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from .core.graph_operations import GraphManager
from .core.algorithms import GraphAlgorithms

logger = logging.getLogger(__name__)

# MCP Protocol Version
MCP_PROTOCOL_VERSION = "2024-11-05"

@dataclass
class MCPRequest:
    """MCP JSON-RPC request message."""
    jsonrpc: str = "2.0"
    id: int = 0
    method: str = ""
    params: Optional[Dict[str, Any]] = None

@dataclass
class MCPResponse:
    """MCP JSON-RPC response message."""
    jsonrpc: str = "2.0"
    id: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

@dataclass
class MCPNotification:
    """MCP JSON-RPC notification message."""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None

class NetworkXMCPServer:
    """Minimal MCP server for NetworkX graph operations."""
    
    def __init__(self):
        self.graph_manager = GraphManager()
        self.algorithms = GraphAlgorithms()
        self.client_capabilities = {}
        self.initialized = False
        
    async def start_stdio_server(self):
        """Start the MCP server using stdio transport."""
        logger.info("Starting NetworkX MCP Server (Minimal) via stdio")
        
        # Read messages from stdin and write responses to stdout
        while True:
            try:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                # Parse JSON-RPC message
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    continue
                
                # Process message
                response = await self.handle_message(message)
                
                # Send response if it's a request (has id)
                if response and "id" in message:
                    response_json = json.dumps(response, separators=(',', ':'))
                    print(response_json, flush=True)
                    
            except KeyboardInterrupt:
                logger.info("Server interrupted by user")
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                break
                
        logger.info("Server stopped")
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming MCP message."""
        try:
            # Validate JSON-RPC format
            if message.get("jsonrpc") != "2.0":
                return self.create_error_response(
                    message.get("id"), -32600, "Invalid Request"
                )
            
            method = message.get("method")
            if not method:
                return self.create_error_response(
                    message.get("id"), -32600, "Missing method"
                )
            
            # Route to appropriate handler
            if method == "initialize":
                return await self.handle_initialize(message)
            elif method == "initialized":
                return await self.handle_initialized(message)
            elif method == "tools/list":
                return await self.handle_tools_list(message)
            elif method == "tools/call":
                return await self.handle_tools_call(message)
            else:
                return self.create_error_response(
                    message.get("id"), -32601, f"Method not found: {method}"
                )
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return self.create_error_response(
                message.get("id"), -32603, f"Internal error: {str(e)}"
            )
    
    def create_error_response(self, id: Optional[int], code: int, message: str) -> Dict[str, Any]:
        """Create JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    async def handle_initialize(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        params = message.get("params", {})
        
        # Validate protocol version
        protocol_version = params.get("protocolVersion")
        if protocol_version != MCP_PROTOCOL_VERSION:
            logger.warning(f"Client protocol version {protocol_version} != {MCP_PROTOCOL_VERSION}")
        
        # Store client capabilities
        self.client_capabilities = params.get("capabilities", {})
        
        # Return server capabilities
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {
                        "listChanged": False  # We don't support dynamic tool changes yet
                    }
                },
                "serverInfo": {
                    "name": "networkx-mcp-server",
                    "version": "1.0.0"
                }
            }
        }
    
    async def handle_initialized(self, message: Dict[str, Any]) -> None:
        """Handle MCP initialized notification."""
        self.initialized = True
        logger.info("MCP client initialized successfully")
        return None  # Notifications don't return responses
    
    async def handle_tools_list(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request - return available graph tools."""
        if not self.initialized:
            return self.create_error_response(
                message["id"], -32002, "Server not initialized"
            )
        
        tools = [
            {
                "name": "create_graph",
                "description": "Create a new graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Unique identifier for the graph"},
                        "directed": {"type": "boolean", "description": "Whether the graph is directed", "default": False}
                    },
                    "required": ["graph_id"]
                }
            },
            {
                "name": "add_nodes",
                "description": "Add nodes to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Graph identifier"},
                        "nodes": {"type": "array", "items": {"type": "string"}, "description": "List of node identifiers"}
                    },
                    "required": ["graph_id", "nodes"]
                }
            },
            {
                "name": "add_edges", 
                "description": "Add edges to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Graph identifier"},
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "string"}
                            },
                            "description": "List of edges as [source, target] pairs"
                        }
                    },
                    "required": ["graph_id", "edges"]
                }
            },
            {
                "name": "get_graph_info",
                "description": "Get information about a graph",
                "inputSchema": {
                    "type": "object", 
                    "properties": {
                        "graph_id": {"type": "string", "description": "Graph identifier"}
                    },
                    "required": ["graph_id"]
                }
            },
            {
                "name": "shortest_path",
                "description": "Find shortest path between two nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Graph identifier"},
                        "source": {"type": "string", "description": "Source node"},
                        "target": {"type": "string", "description": "Target node"}
                    },
                    "required": ["graph_id", "source", "target"]
                }
            },
            {
                "name": "centrality_measures",
                "description": "Calculate centrality measures for graph nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Graph identifier"},
                        "measures": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["degree", "betweenness", "closeness", "eigenvector"]},
                            "description": "List of centrality measures to calculate"
                        }
                    },
                    "required": ["graph_id", "measures"]
                }
            },
            {
                "name": "delete_graph",
                "description": "Delete a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Graph identifier"}
                    },
                    "required": ["graph_id"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": tools
            }
        }
    
    async def handle_tools_call(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request - execute graph operations."""
        if not self.initialized:
            return self.create_error_response(
                message["id"], -32002, "Server not initialized"
            )
        
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            # Route to appropriate tool handler
            if tool_name == "create_graph":
                result = await self.tool_create_graph(arguments)
            elif tool_name == "add_nodes":
                result = await self.tool_add_nodes(arguments)
            elif tool_name == "add_edges":
                result = await self.tool_add_edges(arguments)
            elif tool_name == "get_graph_info":
                result = await self.tool_get_graph_info(arguments)
            elif tool_name == "shortest_path":
                result = await self.tool_shortest_path(arguments)
            elif tool_name == "centrality_measures":
                result = await self.tool_centrality_measures(arguments)
            elif tool_name == "delete_graph":
                result = await self.tool_delete_graph(arguments)
            else:
                return self.create_error_response(
                    message["id"], -32601, f"Unknown tool: {tool_name}"
                )
            
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return self.create_error_response(
                message["id"], -32603, f"Tool execution failed: {str(e)}"
            )
    
    # Tool implementations
    async def tool_create_graph(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new graph."""
        graph_id = args["graph_id"]
        directed = args.get("directed", False)
        
        result = self.graph_manager.create_graph(graph_id, directed=directed)
        return result
    
    async def tool_add_nodes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add nodes to a graph."""
        graph_id = args["graph_id"]
        nodes = args["nodes"]
        
        result = self.graph_manager.add_nodes_from(graph_id, nodes)
        return result
    
    async def tool_add_edges(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add edges to a graph."""
        graph_id = args["graph_id"]
        edges = args["edges"]
        
        # Convert edge list to tuples
        edge_tuples = [(edge[0], edge[1]) for edge in edges]
        result = self.graph_manager.add_edges_from(graph_id, edge_tuples)
        return result
    
    async def tool_get_graph_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get graph information."""
        graph_id = args["graph_id"]
        
        result = self.graph_manager.get_graph_info(graph_id)
        return result
    
    async def tool_shortest_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Find shortest path between nodes."""
        graph_id = args["graph_id"]
        source = args["source"]
        target = args["target"]
        
        try:
            graph = self.graph_manager.get_graph(graph_id)
            path = self.algorithms.shortest_path(graph, source, target)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def tool_centrality_measures(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate centrality measures."""
        graph_id = args["graph_id"]
        measures = args["measures"]
        
        try:
            graph = self.graph_manager.get_graph(graph_id)
            
            results = {}
            for measure in measures:
                if measure == "degree":
                    results["degree"] = dict(nx.degree_centrality(graph))
                elif measure == "betweenness":
                    results["betweenness"] = dict(nx.betweenness_centrality(graph))
                elif measure == "closeness":
                    results["closeness"] = dict(nx.closeness_centrality(graph))
                elif measure == "eigenvector":
                    try:
                        results["eigenvector"] = dict(nx.eigenvector_centrality(graph))
                    except nx.NetworkXException:
                        results["eigenvector"] = "Could not compute (graph may not be connected)"
                        
            return {"success": True, "centrality": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def tool_delete_graph(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a graph."""
        graph_id = args["graph_id"]
        
        result = self.graph_manager.delete_graph(graph_id)
        return result


def main():
    """Main entry point for minimal MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr  # Log to stderr to avoid interfering with stdio protocol
    )
    
    server = NetworkXMCPServer()
    
    try:
        # Run the stdio server
        asyncio.run(server.start_stdio_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()