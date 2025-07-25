#!/usr/bin/env python3
"""
NetworkX MCP Server - Refactored and Modular
Core server functionality with plugin-based architecture.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

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
)
from .core.basic_operations import (
    add_nodes as _add_nodes,
)
from .core.basic_operations import (
    betweenness_centrality as _betweenness_centrality,
)
from .core.basic_operations import (
    community_detection as _community_detection,
)
from .core.basic_operations import (
    connected_components as _connected_components,
)
from .core.basic_operations import (
    create_graph as _create_graph,
)
from .core.basic_operations import (
    degree_centrality as _degree_centrality,
)
from .core.basic_operations import (
    export_json as _export_json,
)
from .core.basic_operations import (
    get_graph_info as _get_graph_info,
)
from .core.basic_operations import (
    import_csv as _import_csv,
)
from .core.basic_operations import (
    pagerank as _pagerank,
)
from .core.basic_operations import (
    shortest_path as _shortest_path,
)
from .core.basic_operations import (
    visualize_graph as _visualize_graph,
)

# Global state - simple and effective
graphs: Dict[str, nx.Graph] = {}


class GraphManager:
    """Simple graph manager for test compatibility."""

    def __init__(self) -> None:
        self.graphs = graphs

    def get_graph(self, graph_id: str) -> nx.Graph | None:
        """Get a graph by ID."""
        return graphs.get(graph_id)

    def delete_graph(self, graph_id: str) -> None:
        """Delete a graph by ID."""
        if graph_id in graphs:
            del graphs[graph_id]


# Create global graph manager instance
graph_manager = GraphManager()

# Optional authentication
try:
    from .auth import APIKeyManager, AuthMiddleware

    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False

# Optional monitoring
try:
    from .monitoring import HealthMonitor

    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False


# Re-export functions with graphs parameter bound
def create_graph(name: str, directed: bool = False) -> Any:
    return _create_graph(name, directed, graphs)


def add_nodes(graph_name: str, nodes: List) -> Any:
    return _add_nodes(graph_name, nodes, graphs)


def add_edges(graph_name: str, edges: List) -> Any:
    return _add_edges(graph_name, edges, graphs)


def get_graph_info(graph_name: str) -> Any:
    return _get_graph_info(graph_name, graphs)


def shortest_path(graph_name: str, source: Any, target: Any) -> Any:
    return _shortest_path(graph_name, source, target, graphs)


def degree_centrality(graph_name: str) -> Any:
    return _degree_centrality(graph_name, graphs)


def betweenness_centrality(graph_name: str) -> Any:
    return _betweenness_centrality(graph_name, graphs)


def connected_components(graph_name: str) -> Any:
    return _connected_components(graph_name, graphs)


def pagerank(graph_name: str) -> Any:
    return _pagerank(graph_name, graphs)


def visualize_graph(graph_name: str, layout: str = "spring") -> Any:
    return _visualize_graph(graph_name, layout, graphs)


def import_csv(graph_name: str, csv_data: str, directed: bool = False) -> Any:
    return _import_csv(graph_name, csv_data, directed, graphs)


def export_json(graph_name: str) -> Any:
    return _export_json(graph_name, graphs)


def delete_graph(graph_name: str) -> Any:
    """Delete a graph - compatibility function."""
    if graph_name not in graphs:
        return {"success": False, "error": f"Graph '{graph_name}' not found"}

    del graphs[graph_name]
    return {"success": True, "graph_id": graph_name, "deleted": True}


def community_detection(graph_name: str) -> Any:
    return _community_detection(graph_name, graphs)


class NetworkXMCPServer:
    """Minimal MCP server - no unnecessary abstraction."""

    def __init__(
        self, auth_required: bool = True, enable_monitoring: bool = False
    ) -> None:
        self.running = True
        self.mcp = self  # For test compatibility
        self.graphs = graphs  # Reference to global graphs

        # Set up authentication if enabled
        self.auth_required = auth_required and HAS_AUTH
        if self.auth_required:
            self.key_manager = APIKeyManager()
            self.auth = AuthMiddleware(self.key_manager, required=auth_required)
        else:
            self.auth = None
            # SECURITY WARNING: Authentication is disabled
            import warnings

            warnings.warn(
                "SECURITY WARNING: Authentication is disabled! "
                "This allows unrestricted access to all server functionality. "
                "Enable authentication in production with auth_required=True.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Set up monitoring if enabled
        self.monitoring_enabled = enable_monitoring and HAS_MONITORING
        if self.monitoring_enabled:
            self.monitor = HealthMonitor()
            self.monitor.graphs = graphs  # Give monitor access to graphs
        else:
            self.monitor = None

    def tool(self, func: Any) -> Any:
        """Mock tool decorator for test compatibility."""
        return func

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route requests to handlers."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        # Handle authentication if required
        auth_data = None
        if self.auth and method not in ["initialize", "initialized"]:
            try:
                auth_data = self.auth.authenticate(request)
                # Remove API key from params to avoid exposing it
                if "api_key" in params:
                    del params["api_key"]
                if "apiKey" in params:
                    del params["apiKey"]
            except ValueError as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                }

        # Record request for monitoring
        if self.monitor:
            self.monitor.record_request(method)

        # Route to appropriate handler
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "networkx-minimal"},
            }
        elif method == "initialized":
            # This is a notification, no response needed
            if req_id is None:
                return None
            result = {}  # Just acknowledge
        elif method == "tools/list[Any]":
            result = {"tools": self._get_tools()}
        elif method == "tools/call":
            # Check permissions for write operations
            if auth_data and self.auth:
                tool_name = params.get("name", "")
                write_tools = [
                    "create_graph",
                    "add_nodes",
                    "add_edges",
                    "import_csv",
                    "build_citation_network",
                ]
                if tool_name in write_tools and not self.auth.check_permission(
                    auth_data, "write"
                ):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32603,
                            "message": "Permission denied: write access required",
                        },
                    }
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
        tools = [
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
                "description": "Import graph from CSV edge list[Any] (format: source,target per line)",
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

        # Add health endpoint if monitoring is enabled
        if self.monitoring_enabled and self.monitor:
            tools.append(
                {
                    "name": "health_status",
                    "description": "Get server health and performance metrics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
            )

        return tools

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
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
                        f"Graph '{graph_name}' not found. Available graphs: {list[Any](graphs.keys())}"
                    )
                graph = graphs[graph_name]
                graph.add_nodes_from(args["nodes"])
                result = {"added": len(args["nodes"]), "total": graph.number_of_nodes()}

            elif tool_name == "add_edges":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list[Any](graphs.keys())}"
                    )
                graph = graphs[graph_name]
                edges = [tuple[Any, ...](e) for e in args["edges"]]
                graph.add_edges_from(edges)
                result = {"added": len(edges), "total": graph.number_of_edges()}

            elif tool_name == "shortest_path":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list[Any](graphs.keys())}"
                    )
                graph = graphs[graph_name]
                path = nx.shortest_path(graph, args["source"], args["target"])
                result = {"path": path, "length": len(path) - 1}

            elif tool_name == "get_info":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(
                        f"Graph '{graph_name}' not found. Available graphs: {list[Any](graphs.keys())}"
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
                viz_result = visualize_graph(args["graph"], layout)
                # Rename 'image' key to 'visualization' for backward compatibility
                result = {
                    "visualization": viz_result["image"],
                    "format": viz_result["format"],
                    "layout": viz_result["layout"],
                }

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
                result = analyze_author_impact(
                    args["graph"], args["author_name"], graphs
                )

            elif tool_name == "find_collaboration_patterns":
                result = find_collaboration_patterns(args["graph"], graphs)

            elif tool_name == "detect_research_trends":
                result = detect_research_trends(
                    args["graph"], args.get("time_window", 5), graphs
                )

            elif tool_name == "export_bibtex":
                result = export_bibtex(args["graph"], graphs)

            elif tool_name == "recommend_papers":
                # Handle alternative parameter names for backward compatibility
                seed = args.get("seed_doi") or args.get("seed_paper")
                max_recs = args.get("max_recommendations") or args.get("top_n", 10)

                if not seed:
                    raise ValueError(
                        "Missing required parameter: seed_doi or seed_paper"
                    )

                result = recommend_papers(args["graph"], seed, max_recs, graphs)

            elif tool_name == "resolve_doi":
                result = resolve_doi(args["doi"])
                if result is None:
                    raise ValueError(f"Could not resolve DOI: {args['doi']}")

            elif tool_name == "health_status":
                if self.monitor:
                    result = self.monitor.get_health_status()
                else:
                    result = {"status": "monitoring_disabled"}

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            }

    async def run(self) -> None:
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
                if response is not None:
                    print(json.dumps(response), flush=True)

            except Exception as e:
                print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)


# Create module-level mcp instance for test compatibility
mcp = NetworkXMCPServer()


def main() -> None:
    """Main entry point for the NetworkX MCP Server."""
    # Check environment variables
    auth_required = os.environ.get("NETWORKX_MCP_AUTH", "true").lower() == "true"
    enable_monitoring = (
        os.environ.get("NETWORKX_MCP_MONITORING", "false").lower() == "true"
    )

    import logging

    logging.basicConfig(level=logging.INFO)

    if auth_required:
        logging.info(
            "✅ SECURE: Starting NetworkX MCP Server with authentication ENABLED"
        )
        logging.info(
            "Use 'python -m networkx_mcp.auth generate <name>' to create API keys"
        )
    else:
        logging.warning("🚨 SECURITY ALERT: Authentication is DISABLED!")
        logging.warning("This allows unrestricted access to all server functionality!")
        logging.warning("To enable security: export NETWORKX_MCP_AUTH=true")
        logging.warning("Or set auth_required=True in server constructor")

        # Require explicit confirmation to run without auth
        if not os.environ.get("NETWORKX_MCP_INSECURE_CONFIRM", "").lower() == "true":
            logging.error("SECURITY: Server startup blocked for safety")
            logging.error(
                "To run without auth (NOT RECOMMENDED): export NETWORKX_MCP_INSECURE_CONFIRM=true"
            )
            raise RuntimeError(
                "SECURITY: Authentication disabled but not explicitly confirmed. "
                "Set NETWORKX_MCP_INSECURE_CONFIRM=true to bypass this safety check."
            )

    if enable_monitoring:
        logging.info("Starting NetworkX MCP Server with monitoring enabled")
        logging.info("Health status available via 'health_status' tool")

    server = NetworkXMCPServer(
        auth_required=auth_required, enable_monitoring=enable_monitoring
    )
    asyncio.run(server.run())


# Run the server
if __name__ == "__main__":
    main()
