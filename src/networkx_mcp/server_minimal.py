#!/usr/bin/env python3
"""
Actually Minimal NetworkX MCP Server
Only 150 lines. No BS. Just works.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use("Agg")  # Use non-interactive backend
import base64
import csv
import io

import networkx.algorithms.community as nx_comm

# Global state - simple and effective
graphs: Dict[str, nx.Graph] = {}


# Compatibility exports for tests
def create_graph(name: str, directed: bool = False):
    """Create a graph - compatibility function."""
    graphs[name] = nx.DiGraph() if directed else nx.Graph()
    return {"created": name, "type": "directed" if directed else "undirected"}


def add_nodes(graph_name: str, nodes: List):
    """Add nodes - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    graph.add_nodes_from(nodes)
    return {"added": len(nodes), "total": graph.number_of_nodes()}


def add_edges(graph_name: str, edges: List):
    """Add edges - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    edge_tuples = [tuple(e) for e in edges]
    graph.add_edges_from(edge_tuples)
    return {"added": len(edge_tuples), "total": graph.number_of_edges()}


def get_graph_info(graph_name: str):
    """Get graph info - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
    }


def shortest_path(graph_name: str, source, target):
    """Find shortest path - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    path = nx.shortest_path(graph, source, target)
    return {"path": path, "length": len(path) - 1}


def degree_centrality(graph_name: str):
    """Calculate degree centrality - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    centrality = nx.degree_centrality(graph)
    # Convert to serializable format and sort by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return {
        "centrality": dict(sorted_nodes[:10]),  # Top 10 nodes
        "most_central": sorted_nodes[0] if sorted_nodes else None,
    }


def betweenness_centrality(graph_name: str):
    """Calculate betweenness centrality - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    centrality = nx.betweenness_centrality(graph)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return {
        "centrality": dict(sorted_nodes[:10]),  # Top 10 nodes
        "most_central": sorted_nodes[0] if sorted_nodes else None,
    }


def connected_components(graph_name: str):
    """Find connected components - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    # Convert sets to lists for JSON serialization
    components_list = [list(comp) for comp in components]
    components_list.sort(key=len, reverse=True)  # Largest first

    return {
        "num_components": len(components_list),
        "component_sizes": [len(comp) for comp in components_list],
        "largest_component": components_list[0] if components_list else [],
    }


def pagerank(graph_name: str):
    """Calculate PageRank - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    pr = nx.pagerank(graph)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return {
        "pagerank": dict(sorted_nodes[:10]),  # Top 10 nodes
        "highest_rank": sorted_nodes[0] if sorted_nodes else None,
    }


def visualize_graph(graph_name: str, layout: str = "spring"):
    """Visualize graph and return as base64 image - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    plt.figure(figsize=(10, 8))

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph)

    # Draw the graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        arrows=True,
    )

    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close()

    # Convert to base64
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "image": f"data:image/png;base64,{image_base64}",
        "format": "png",
        "layout": layout,
    }


def import_csv(graph_name: str, csv_data: str, directed: bool = False):
    """Import graph from CSV edge list - compatibility function."""
    # Parse CSV data
    reader = csv.reader(io.StringIO(csv_data))
    edges = []

    for row in reader:
        if len(row) >= 2:
            # Handle both numeric and string nodes
            try:
                source = int(row[0])
            except:
                source = row[0].strip()
            try:
                target = int(row[1])
            except:
                target = row[1].strip()

            edges.append((source, target))

    # Create graph
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    graphs[graph_name] = graph

    return {
        "imported": graph_name,
        "type": "directed" if directed else "undirected",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }


def export_json(graph_name: str):
    """Export graph as JSON - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Convert to node-link format
    data = nx.node_link_data(graph)

    return {
        "graph_data": data,
        "format": "node-link",
        "nodes": len(data["nodes"]),
        "edges": len(data["links"]),
    }


def community_detection(graph_name: str):
    """Detect communities in the graph - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Use Louvain method for community detection
    communities = nx_comm.louvain_communities(graph)

    # Convert to list format
    communities_list = [list(comm) for comm in communities]
    communities_list.sort(key=len, reverse=True)  # Largest first

    # Create node to community mapping
    node_community = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            node_community[node] = i

    return {
        "num_communities": len(communities_list),
        "community_sizes": [len(comm) for comm in communities_list],
        "largest_community": communities_list[0] if communities_list else [],
        "node_community_map": dict(list(node_community.items())[:20]),  # First 20 nodes
    }


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


# Run the server
if __name__ == "__main__":
    server = MinimalMCPServer()
    asyncio.run(server.run())
