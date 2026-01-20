#!/usr/bin/env python3
"""
NetworkX MCP Server - Refactored and Modular
Core server functionality with plugin-based architecture.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import networkx as nx

from .errors import ErrorCodes, MCPError

logger = logging.getLogger(__name__)

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
# Import the new thread-safe graph cache with memory management
from .graph_cache import graphs


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
    from .monitoring_legacy import HealthMonitor

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
        self,
        auth_required: bool = False,
        enable_monitoring: bool = False,  # Changed default to False for MCP
    ) -> None:
        self.running = True
        self.initialized = False  # Track initialization state
        self.mcp = self  # For test compatibility
        self.graphs = graphs  # Reference to global graphs

        # Set up authentication if enabled
        self.auth_required = auth_required and HAS_AUTH
        if self.auth_required:
            self.key_manager = APIKeyManager()
            self.auth = AuthMiddleware(self.key_manager, required=auth_required)
        else:
            self.auth = None
            # Only show warning if not in test mode
            if not os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
                "NETWORKX_MCP_SUPPRESS_AUTH_WARNING"
            ):
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

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
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
            self.initialized = True  # Mark as initialized
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {"listChangedSupport": False},
                    "prompts": {},
                },
                "serverInfo": {"name": "networkx-mcp-server", "version": "1.0.0"},
            }
        elif method == "initialized":
            # This is a notification, no response needed
            if req_id is None:
                return None
            result = {}  # Just acknowledge
        elif method == "tools/list":
            # Check if initialized (except for init methods)
            if not self.initialized:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32002, "message": "Server not initialized"},
                }
            result = {"tools": self._get_tools()}
        elif method == "tools/call":
            # Check permissions for write operations
            if auth_data and self.auth:
                tool_name = params.get("name", "")
                write_tools = [
                    "create_graph",
                    "add_nodes",
                    "add_edges",
                    "delete_graph",
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
            # Check if initialized
            if not self.initialized:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32002, "message": "Server not initialized"},
                }
            result = await self._call_tool(params)
        elif method == "resources/list":
            # MCP Resources API - return empty list for now
            result = {"resources": []}
        elif method == "resources/read":
            # MCP Resources API - not implemented yet
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "resources/read not implemented"},
            }
        elif method == "prompts/list":
            # MCP Prompts API - return empty list for now
            result = {"prompts": []}
        elif method == "prompts/get":
            # MCP Prompts API - not implemented yet
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "prompts/get not implemented"},
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle MCP message (alias for handle_request for compatibility).

        Args:
            message: The message to handle

        Returns:
            Response dictionary or None for notifications
        """
        # For notifications (no 'id' field), handle_request returns None
        return await self.handle_request(message)

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
                "description": "Import graph from CSV edge List[Any] (format: source,target per line)",
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
                "name": "delete_graph",
                "description": "Delete a graph from storage",
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

        # Add CI/CD control tools
        cicd_tools = [
            {
                "name": "trigger_workflow",
                "description": "Trigger a GitHub Actions workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow": {
                            "type": "string",
                            "description": "Workflow file name",
                        },
                        "branch": {"type": "string", "default": "main"},
                        "inputs": {
                            "type": "string",
                            "description": "JSON string of inputs",
                        },
                    },
                    "required": ["workflow"],
                },
            },
            {
                "name": "get_workflow_status",
                "description": "Get CI/CD workflow status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "Optional run ID"},
                    },
                },
            },
            {
                "name": "cancel_workflow",
                "description": "Cancel a running workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                    },
                    "required": ["run_id"],
                },
            },
            {
                "name": "rerun_failed_jobs",
                "description": "Rerun failed jobs in a workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                    },
                    "required": ["run_id"],
                },
            },
            {
                "name": "get_dora_metrics",
                "description": "Get DORA metrics for CI/CD performance",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "analyze_workflow_failures",
                "description": "Analyze workflow failures with AI-powered insights",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                    },
                    "required": ["run_id"],
                },
            },
        ]

        # Add CI/CD tools if available
        try:
            # Test if tools module is available
            import importlib.util

            if importlib.util.find_spec("networkx_mcp.tools") is not None:
                tools.extend(cicd_tools)
        except ImportError:
            pass  # CI/CD tools not available

        return tools

    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
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

            elif tool_name == "delete_graph":
                result = delete_graph(args["graph"])

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
                result, error = resolve_doi(args["doi"])
                if result is None:
                    error_msg = error or "Unknown error"
                    raise ValueError(
                        f"Could not resolve DOI: {args['doi']} - {error_msg}"
                    )

            elif tool_name == "health_status":
                if self.monitor:
                    result = self.monitor.get_health_status()
                else:
                    result = {"status": "monitoring_disabled"}

            # CI/CD Control Tools
            elif tool_name == "trigger_workflow":
                try:
                    from .tools import mcp_trigger_workflow

                    result = await mcp_trigger_workflow(
                        args["workflow"], args.get("branch", "main"), args.get("inputs")
                    )
                except ImportError:
                    result = {"error": "CI/CD tools not available"}

            elif tool_name == "get_workflow_status":
                try:
                    from .tools import mcp_get_workflow_status

                    result = await mcp_get_workflow_status(args.get("run_id"))
                except ImportError:
                    result = {"error": "CI/CD tools not available"}

            elif tool_name == "cancel_workflow":
                try:
                    from .tools import mcp_cancel_workflow

                    result = await mcp_cancel_workflow(args["run_id"])
                except ImportError:
                    result = {"error": "CI/CD tools not available"}

            elif tool_name == "rerun_failed_jobs":
                try:
                    from .tools import mcp_rerun_failed_jobs

                    result = await mcp_rerun_failed_jobs(args["run_id"])
                except ImportError:
                    result = {"error": "CI/CD tools not available"}

            elif tool_name == "get_dora_metrics":
                try:
                    from .tools import mcp_get_dora_metrics

                    result = await mcp_get_dora_metrics()
                except ImportError:
                    result = {"error": "CI/CD tools not available"}

            elif tool_name == "analyze_workflow_failures":
                try:
                    from .tools import mcp_analyze_failures

                    result = await mcp_analyze_failures(args["run_id"])
                except ImportError:
                    result = {"error": "CI/CD tools not available"}

            else:
                # Return proper error for unknown tool
                return {
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        except MCPError as e:
            # Return proper JSON-RPC error for MCP-specific errors
            return {"error": e.to_dict()}

        except nx.NetworkXError as e:
            # NetworkX algorithm errors
            logger.warning(f"NetworkX error in tool {tool_name}: {e}")
            return {
                "error": {
                    "code": ErrorCodes.ALGORITHM_ERROR,
                    "message": f"Graph operation failed: {str(e)}",
                }
            }

        except KeyError as e:
            # Missing required parameters
            return {
                "error": {
                    "code": ErrorCodes.INVALID_PARAMS,
                    "message": f"Missing required parameter: {str(e)}",
                }
            }

        except (TypeError, ValueError) as e:
            # Invalid parameter types or values
            return {
                "error": {
                    "code": ErrorCodes.INVALID_PARAMS,
                    "message": f"Invalid parameter: {str(e)}",
                }
            }

        except Exception as e:
            # Unexpected errors - log for debugging
            logger.exception(f"Unexpected error in tool {tool_name}")
            return {
                "error": {
                    "code": ErrorCodes.INTERNAL_ERROR,
                    "message": f"Internal error: {str(e)}",
                }
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

            except json.JSONDecodeError as e:
                # Invalid JSON input
                logger.error(f"Invalid JSON input: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": ErrorCodes.PARSE_ERROR,
                        "message": f"Parse error: {str(e)}",
                    },
                    "id": None,
                }
                print(json.dumps(error_response), file=sys.stderr, flush=True)

            except (IOError, OSError) as e:
                # IO errors (stdin/stdout issues)
                logger.error(f"IO error in main loop: {e}")
                break  # Exit the loop on IO errors

            except Exception as e:
                # Unexpected errors - log and continue
                logger.exception("Unexpected error in main loop")
                print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)


# Create module-level mcp instance for test compatibility
# Suppress auth warning for this default instance
_original_env = os.environ.get("NETWORKX_MCP_SUPPRESS_AUTH_WARNING")
os.environ["NETWORKX_MCP_SUPPRESS_AUTH_WARNING"] = "1"
mcp = NetworkXMCPServer()
# Restore original env value
if _original_env is None:
    del os.environ["NETWORKX_MCP_SUPPRESS_AUTH_WARNING"]
else:
    os.environ["NETWORKX_MCP_SUPPRESS_AUTH_WARNING"] = _original_env


def main() -> None:
    """Main entry point for the NetworkX MCP Server."""
    # Check environment variables - default to False for MCP compatibility
    auth_required = os.environ.get("NETWORKX_MCP_AUTH", "false").lower() == "true"
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
        # Check if we're in production mode
        is_production = (
            os.environ.get("NETWORKX_MCP_ENV", "development").lower() == "production"
        )

        if is_production:
            logging.warning(
                "⚠️ SECURITY WARNING: Authentication is DISABLED in production!"
            )
            logging.warning(
                "This allows unrestricted access to all server functionality!"
            )
            logging.warning("To enable security: export NETWORKX_MCP_AUTH=true")

            # Require explicit confirmation to run without auth in production
            if (
                not os.environ.get("NETWORKX_MCP_INSECURE_CONFIRM", "").lower()
                == "true"
            ):
                logging.error("SECURITY: Production server startup blocked for safety")
                logging.error(
                    "To run without auth in production: export NETWORKX_MCP_INSECURE_CONFIRM=true"
                )
                raise RuntimeError(
                    "SECURITY: Authentication disabled in production. "
                    "Set NETWORKX_MCP_INSECURE_CONFIRM=true to bypass this safety check."
                )
        else:
            # Development mode - just show info message
            logging.info("ℹ️ Running in development mode without authentication")
            logging.info(
                "For production use, enable auth with: export NETWORKX_MCP_AUTH=true"
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
