#!/usr/bin/env python3
"""
Actually Minimal NetworkX MCP Server
Only 150 lines. No BS. Just works.
"""

import asyncio
import json
import sys
from typing import Dict, Any, List, Optional, Union
import networkx as nx

# Global state - simple and effective
graphs: Dict[str, nx.Graph] = {}

class MinimalMCPServer:
    """Minimal MCP server - no unnecessary abstraction."""
    
    def __init__(self):
        self.running = True
    
    async def handle_request(self, request: dict) -> dict:
        """Route requests to handlers."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")
        
        # Route to appropriate handler
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "networkx-minimal"}}
        elif method == "initialized":
            result = {}  # Just acknowledge
        elif method == "tools/list":
            result = {"tools": self._get_tools()}
        elif method == "tools/call":
            result = await self._call_tool(params)
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}
        
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        return [
            {
                "name": "create_graph",
                "description": "Create a new graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "directed": {"type": "boolean", "default": False}
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "add_nodes",
                "description": "Add nodes to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "nodes": {"type": "array", "items": {"type": ["string", "number"]}}
                    },
                    "required": ["graph", "nodes"]
                }
            },
            {
                "name": "add_edges",
                "description": "Add edges to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "edges": {"type": "array", "items": {"type": "array", "items": {"type": ["string", "number"]}}}
                    },
                    "required": ["graph", "edges"]
                }
            },
            {
                "name": "shortest_path",
                "description": "Find shortest path between nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "source": {"type": ["string", "number"]},
                        "target": {"type": ["string", "number"]}
                    },
                    "required": ["graph", "source", "target"]
                }
            },
            {
                "name": "get_info",
                "description": "Get graph information",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            }
        ]
    
    async def _call_tool(self, params: dict) -> dict:
        """Execute a tool."""
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        try:
            if tool_name == "create_graph":
                name = args["name"]
                directed = args.get("directed", False)
                graphs[name] = nx.DiGraph() if directed else nx.Graph()
                result = {"created": name, "type": "directed" if directed else "undirected"}
                
            elif tool_name == "add_nodes":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                graph.add_nodes_from(args["nodes"])
                result = {"added": len(args["nodes"]), "total": graph.number_of_nodes()}
                
            elif tool_name == "add_edges":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                edges = [tuple(e) for e in args["edges"]]
                graph.add_edges_from(edges)
                result = {"added": len(edges), "total": graph.number_of_edges()}
                
            elif tool_name == "shortest_path":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                path = nx.shortest_path(graph, args["source"], args["target"])
                result = {"path": path, "length": len(path) - 1}
                
            elif tool_name == "get_info":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                result = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "directed": graph.is_directed()
                }
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}
    
    async def run(self):
        """Main server loop - read stdin, write stdout."""
        while self.running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
                
            except Exception as e:
                print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)

# Run the server
if __name__ == "__main__":
    server = MinimalMCPServer()
    asyncio.run(server.run())