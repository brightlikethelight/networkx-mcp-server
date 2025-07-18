#!/usr/bin/env python3
"""
NetworkX MCP Server - Refactored and Modular
Core server functionality with plugin-based architecture.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

import networkx as nx

# Import academic functions from plugin
from .academic import (
    analyze_author_impact,
    build_citation_network,
    detect_research_trends,
    export_bibtex,
    find_collaboration_patterns,
    recommend_papers,
    resolve_doi,
)

# Import basic operations
from .core.basic_operations import (
    add_edges as _add_edges,
    add_nodes as _add_nodes,
    betweenness_centrality as _betweenness_centrality,
    community_detection as _community_detection,
    connected_components as _connected_components,
    create_graph as _create_graph,
    degree_centrality as _degree_centrality,
    export_json as _export_json,
    get_graph_info as _get_graph_info,
    import_csv as _import_csv,
    pagerank as _pagerank,
    shortest_path as _shortest_path,
    visualize_graph as _visualize_graph,
)

# Global state - simple and effective
graphs: Dict[str, nx.Graph] = {}

# Re-export functions with graphs parameter bound
def create_graph(name: str, directed: bool = False):
    return _create_graph(name, directed, graphs)

def add_nodes(graph_name: str, nodes: List):
    return _add_nodes(graph_name, nodes, graphs)

def add_edges(graph_name: str, edges: List):
    return _add_edges(graph_name, edges, graphs)

def get_graph_info(graph_name: str):
    return _get_graph_info(graph_name, graphs)

def shortest_path(graph_name: str, source, target):
    return _shortest_path(graph_name, source, target, graphs)

def degree_centrality(graph_name: str):
    return _degree_centrality(graph_name, graphs)

def betweenness_centrality(graph_name: str):
    return _betweenness_centrality(graph_name, graphs)

def connected_components(graph_name: str):
    return _connected_components(graph_name, graphs)

def pagerank(graph_name: str):
    return _pagerank(graph_name, graphs)

def visualize_graph(graph_name: str, layout: str = "spring"):
    return _visualize_graph(graph_name, layout, graphs)

def import_csv(graph_name: str, csv_data: str, directed: bool = False):
    return _import_csv(graph_name, csv_data, directed, graphs)

def export_json(graph_name: str):
    return _export_json(graph_name, graphs)

def community_detection(graph_name: str):
    return _community_detection(graph_name, graphs)






class NetworkXMCPServer:
    """Minimal MCP server - no unnecessary abstraction."""

    def __init__(self):
        self.running = True
        self.mcp = self  # For test compatibility
        self.graphs = graphs  # Reference to global graphs

    def tool(self, func):
        """Mock tool decorator for test compatibility."""
        return func

    async def handle_request(self, request: dict) -> dict:
        """Route requests to handlers."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        # Route to appropriate handler
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "networkx-minimal"},
            }
        elif method == "initialized":
            result = {}  # Just acknowledge
        elif method == "tools/list":
            result = {"tools": self._get_tools()}
        elif method == "tools/call":
            result = await self._call_tool(params)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

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
                        "directed": {"type": "boolean", "default": False},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "add_nodes",
                "description": "Add nodes to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "nodes": {
                            "type": "array",
                            "items": {"type": ["string", "number"]},
                        },
                    },
                    "required": ["graph", "nodes"],
                },
            },
            {
                "name": "add_edges",
                "description": "Add edges to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "number"]},
                            },
                        },
                    },
                    "required": ["graph", "edges"],
                },
            },
            {
                "name": "shortest_path",
                "description": "Find shortest path between nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "source": {"type": ["string", "number"]},
                        "target": {"type": ["string", "number"]},
                    },
                    "required": ["graph", "source", "target"],
                },
            },
            {
                "name": "get_info",
                "description": "Get graph information",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "degree_centrality",
                "description": "Calculate degree centrality for all nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "betweenness_centrality",
                "description": "Calculate betweenness centrality for all nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "connected_components",
                "description": "Find connected components in the graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "pagerank",
                "description": "Calculate PageRank for all nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "community_detection",
                "description": "Detect communities in the graph using Louvain method",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "visualize_graph",
                "description": "Create a visualization of the graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "layout": {
                            "type": "string",
                            "enum": ["spring", "circular", "kamada_kawai"],
                            "default": "spring",
                        },
                    },
                    "required": ["graph"],
                },
            },
            {
                "name": "import_csv",
                "description": "Import graph from CSV edge list (format: source,target per line)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "csv_data": {"type": "string"},
                        "directed": {"type": "boolean", "default": False},
                    },
                    "required": ["graph", "csv_data"],
                },
            },
            {
                "name": "export_json",
                "description": "Export graph as JSON in node-link format",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "build_citation_network",
                "description": "Build citation network from DOIs using CrossRef API",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "seed_dois": {"type": "array", "items": {"type": "string"}},
                        "max_depth": {"type": "integer", "default": 2},
                    },
                    "required": ["graph", "seed_dois"],
                },
            },
            {
                "name": "analyze_author_impact",
                "description": "Analyze author impact metrics including h-index",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "author_name": {"type": "string"},
                    },
                    "required": ["graph", "author_name"],
                },
            },
            {
                "name": "find_collaboration_patterns",
                "description": "Find collaboration patterns in citation network",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "detect_research_trends",
                "description": "Detect research trends over time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "time_window": {"type": "integer", "default": 5},
                    },
                    "required": ["graph"],
                },
            },
            {
                "name": "export_bibtex",
                "description": "Export citation network as BibTeX format",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
            },
            {
                "name": "recommend_papers",
                "description": "Recommend papers based on citation network analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "seed_doi": {"type": "string"},
                        "max_recommendations": {"type": "integer", "default": 10},
                    },
                    "required": ["graph", "seed_doi"],
                },
            },
            {
                "name": "resolve_doi",
                "description": "Resolve DOI to publication metadata using CrossRef API",
                "inputSchema": {
                    "type": "object",
                    "properties": {"doi": {"type": "string"}},
                    "required": ["doi"],
                },
            },
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
                result = {
                    "created": name,
                    "type": "directed" if directed else "undirected",
                }

            elif tool_name == "add_nodes":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}"
                    )
                graph = graphs[graph_name]
                graph.add_nodes_from(args["nodes"])
                result = {"added": len(args["nodes"]), "total": graph.number_of_nodes()}

            elif tool_name == "add_edges":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}"
                    )
                graph = graphs[graph_name]
                edges = [tuple(e) for e in args["edges"]]
                graph.add_edges_from(edges)
                result = {"added": len(edges), "total": graph.number_of_edges()}

            elif tool_name == "shortest_path":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}"
                    )
                graph = graphs[graph_name]
                path = nx.shortest_path(graph, args["source"], args["target"])
                result = {"path": path, "length": len(path) - 1}

            elif tool_name == "get_info":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}"
                    )
                graph = graphs[graph_name]
                result = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "directed": graph.is_directed(),
                }

            elif tool_name == "degree_centrality":
                result = degree_centrality(args["graph"])

            elif tool_name == "betweenness_centrality":
                result = betweenness_centrality(args["graph"])

            elif tool_name == "connected_components":
                result = connected_components(args["graph"])

            elif tool_name == "pagerank":
                result = pagerank(args["graph"])

            elif tool_name == "community_detection":
                result = community_detection(args["graph"])

            elif tool_name == "visualize_graph":
                layout = args.get("layout", "spring")
                result = visualize_graph(args["graph"], layout)

            elif tool_name == "import_csv":
                result = import_csv(
                    args["graph"], args["csv_data"], args.get("directed", False)
                )

            elif tool_name == "export_json":
                result = export_json(args["graph"])

            elif tool_name == "build_citation_network":
                result = build_citation_network(
                    args["graph"], args["seed_dois"], args.get("max_depth", 2), graphs
                )

            elif tool_name == "analyze_author_impact":
                result = analyze_author_impact(args["graph"], args["author_name"], graphs)

            elif tool_name == "find_collaboration_patterns":
                result = find_collaboration_patterns(args["graph"], graphs)

            elif tool_name == "detect_research_trends":
                result = detect_research_trends(
                    args["graph"], args.get("time_window", 5), graphs
                )

            elif tool_name == "export_bibtex":
                result = export_bibtex(args["graph"], graphs)

            elif tool_name == "recommend_papers":
                result = recommend_papers(
                    args["graph"], args["seed_doi"], args.get("max_recommendations", 10), graphs
                )

            elif tool_name == "resolve_doi":
                result = resolve_doi(args["doi"])
                if result is None:
                    raise ValueError(f"Could not resolve DOI: {args['doi']}")

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            }

    async def run(self):
        """Main server loop - read stdin, write stdout."""
        while self.running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break

                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)

            except Exception as e:
                print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)


# Create module-level mcp instance for test compatibility
mcp = NetworkXMCPServer()


def main():
    """Main entry point for the NetworkX MCP Server."""
    server = NetworkXMCPServer()
    asyncio.run(server.run())


# Run the server
if __name__ == "__main__":
    main()
