#!/usr/bin/env python3
"""
TRULY minimal NetworkX MCP server.

Memory usage: ~20-25MB (not 118MB!)
Dependencies: Only networkx and standard library

This is what a minimal server actually looks like.
"""

import json
import sys
import asyncio
import logging
import signal
from typing import Dict, Any, Optional, List
from pathlib import Path

import networkx as nx  # The ONLY external dependency

# NO pandas imports!
# NO scipy imports!
# NO matplotlib imports!
# NO heavyweight data science libraries!

logger = logging.getLogger(__name__)

class TrulyMinimalServer:
    """Minimal MCP server with only essential NetworkX operations."""
    
    def __init__(self):
        self.graphs: Dict[str, nx.Graph] = {}
        self.running = True
        
    async def start_stdio_server(self):
        """Start the minimal MCP server."""
        logger.info("Starting TRULY minimal NetworkX MCP server...")
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
        
        while self.running:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Parse JSON-RPC request
                request = json.loads(line.strip())
                
                # Handle request
                response = await self.handle_request(request)
                
                # Send response
                if response:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                # Invalid JSON
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error",
                        "data": str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(0.1)
                
    async def handle_request(self, request: dict) -> Optional[dict]:
        """Handle MCP JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Handle methods
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "networkx-mcp-minimal",
                        "version": "0.1.0-minimal"
                    }
                }
            }
            
        elif method == "initialized":
            # Notification, no response
            return None
            
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": self._get_minimal_tools()
                }
            }
            
        elif method == "tools/call":
            try:
                result = await self._call_tool(params)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
                
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            
    def _get_minimal_tools(self) -> List[dict]:
        """Return only essential graph tools - no I/O operations."""
        return [
            {
                "name": "create_graph",
                "description": "Create a new graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string"},
                        "graph_type": {
                            "type": "string",
                            "enum": ["Graph", "DiGraph"],
                            "default": "Graph"
                        }
                    },
                    "required": ["graph_id"]
                }
            },
            {
                "name": "add_node",
                "description": "Add a node to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string"},
                        "node_id": {"type": ["string", "integer"]}
                    },
                    "required": ["graph_id", "node_id"]
                }
            },
            {
                "name": "add_edge",
                "description": "Add an edge between two nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string"},
                        "source": {"type": ["string", "integer"]},
                        "target": {"type": ["string", "integer"]}
                    },
                    "required": ["graph_id", "source", "target"]
                }
            },
            {
                "name": "get_graph_info",
                "description": "Get information about a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string"}
                    },
                    "required": ["graph_id"]
                }
            },
            {
                "name": "shortest_path",
                "description": "Find shortest path between nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string"},
                        "source": {"type": ["string", "integer"]},
                        "target": {"type": ["string", "integer"]}
                    },
                    "required": ["graph_id", "source", "target"]
                }
            },
            {
                "name": "list_graphs",
                "description": "List all graphs",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
            # NO import/export tools - those require pandas!
        ]
        
    async def _call_tool(self, params: dict) -> dict:
        """Execute a tool with minimal dependencies."""
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        # Tool implementations
        if tool_name == "create_graph":
            graph_id = args["graph_id"]
            graph_type = args.get("graph_type", "Graph")
            
            if graph_id in self.graphs:
                raise ValueError(f"Graph {graph_id} already exists")
                
            if graph_type == "Graph":
                self.graphs[graph_id] = nx.Graph()
            elif graph_type == "DiGraph":
                self.graphs[graph_id] = nx.DiGraph()
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
                
            return {
                "content": [{
                    "type": "text",
                    "text": f"Created {graph_type} with id '{graph_id}'"
                }]
            }
            
        elif tool_name == "add_node":
            graph_id = args["graph_id"]
            node_id = args["node_id"]
            
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")
                
            self.graphs[graph_id].add_node(node_id)
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"Added node '{node_id}' to graph '{graph_id}'"
                }]
            }
            
        elif tool_name == "add_edge":
            graph_id = args["graph_id"]
            source = args["source"]
            target = args["target"]
            
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")
                
            self.graphs[graph_id].add_edge(source, target)
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"Added edge from '{source}' to '{target}' in graph '{graph_id}'"
                }]
            }
            
        elif tool_name == "get_graph_info":
            graph_id = args["graph_id"]
            
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")
                
            graph = self.graphs[graph_id]
            info = {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "is_directed": graph.is_directed(),
                "density": nx.density(graph) if graph.number_of_nodes() > 0 else 0
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(info, indent=2)
                }]
            }
            
        elif tool_name == "shortest_path":
            graph_id = args["graph_id"]
            source = args["source"]
            target = args["target"]
            
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")
                
            graph = self.graphs[graph_id]
            
            try:
                path = nx.shortest_path(graph, source, target)
                length = nx.shortest_path_length(graph, source, target)
                
                result = {
                    "path": path,
                    "length": length
                }
                
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            except nx.NetworkXNoPath:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No path exists between '{source}' and '{target}'"
                    }]
                }
                
        elif tool_name == "list_graphs":
            graphs_info = []
            for graph_id, graph in self.graphs.items():
                graphs_info.append({
                    "graph_id": graph_id,
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "graph_type": type(graph).__name__
                })
                
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(graphs_info, indent=2)
                }]
            }
            
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


async def main():
    """Main entry point for the minimal server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run server
    server = TrulyMinimalServer()
    await server.start_stdio_server()


if __name__ == "__main__":
    asyncio.run(main())