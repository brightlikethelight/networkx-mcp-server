#!/usr/bin/env python3
"""
NetworkX MCP Server - Refactored and Modular
Core server functionality with plugin-based architecture.
"""

import asyncio
import inspect
import json
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import networkx as nx

from .__version__ import __version__
from .errors import ErrorCodes, MCPError, validate_graph_id
from .tool_registry import build_registry

logger = logging.getLogger(__name__)

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


# DEPRECATED: use handlers.py directly. Will be removed in v4.0.
# Re-export functions with graphs parameter bound
import warnings

_DEPRECATION_MSG = (
    "{name}() is deprecated, use handlers.py directly. Will be removed in v4.0."
)


def create_graph(name: str, directed: bool = False) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="create_graph"), DeprecationWarning, stacklevel=2
    )
    return _create_graph(name, directed, graphs)


def add_nodes(graph_name: str, nodes: List) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="add_nodes"), DeprecationWarning, stacklevel=2
    )
    return _add_nodes(graph_name, nodes, graphs)


def add_edges(graph_name: str, edges: List) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="add_edges"), DeprecationWarning, stacklevel=2
    )
    return _add_edges(graph_name, edges, graphs)


def get_graph_info(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="get_graph_info"), DeprecationWarning, stacklevel=2
    )
    return _get_graph_info(graph_name, graphs)


def shortest_path(graph_name: str, source: Any, target: Any) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="shortest_path"), DeprecationWarning, stacklevel=2
    )
    return _shortest_path(graph_name, source, target, graphs)


def degree_centrality(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="degree_centrality"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _degree_centrality(graph_name, graphs)


def betweenness_centrality(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="betweenness_centrality"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _betweenness_centrality(graph_name, graphs)


def connected_components(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="connected_components"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _connected_components(graph_name, graphs)


def pagerank(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="pagerank"), DeprecationWarning, stacklevel=2
    )
    return _pagerank(graph_name, graphs)


def visualize_graph(graph_name: str, layout: str = "spring") -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="visualize_graph"),
        DeprecationWarning,
        stacklevel=2,
    )
    return _visualize_graph(graph_name, layout, graphs)


def import_csv(graph_name: str, csv_data: str, directed: bool = False) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="import_csv"), DeprecationWarning, stacklevel=2
    )
    return _import_csv(graph_name, csv_data, directed, graphs)


def export_json(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="export_json"), DeprecationWarning, stacklevel=2
    )
    return _export_json(graph_name, graphs)


def delete_graph(graph_name: str) -> Any:
    """Delete a graph - compatibility function."""
    warnings.warn(
        _DEPRECATION_MSG.format(name="delete_graph"), DeprecationWarning, stacklevel=2
    )
    if graph_name not in graphs:
        return {"success": False, "error": f"Graph '{graph_name}' not found"}

    del graphs[graph_name]
    return {"success": True, "graph_id": graph_name, "deleted": True}


def community_detection(graph_name: str) -> Any:
    warnings.warn(
        _DEPRECATION_MSG.format(name="community_detection"),
        DeprecationWarning,
        stacklevel=2,
    )
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
        # Thread safety note: MCP protocol is request-response (one active
        # request at a time via await), so concurrent handler execution cannot
        # occur. The executor prevents blocking the asyncio event loop on
        # CPU-bound NetworkX operations. Individual graph objects are NOT
        # thread-safe, but sequential MCP dispatch makes this safe.
        self._executor = ThreadPoolExecutor(max_workers=4)

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

        # Build tool registry (single source of truth for tools)
        self._registry = build_registry(
            monitoring_enabled=self.monitoring_enabled,
            monitor=self.monitor,
        )

    def tool(self, func: Any) -> Any:
        """Mock tool decorator for test compatibility."""
        return func

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route requests to handlers."""
        # JSON-RPC 2.0 validation
        if request.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": ErrorCodes.INVALID_REQUEST,
                    "message": "Invalid Request: missing or wrong 'jsonrpc' version (must be '2.0')",
                },
            }

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
                "serverInfo": {"name": "networkx-mcp-server", "version": __version__},
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
                    "error": {
                        "code": ErrorCodes.SERVER_NOT_INITIALIZED,
                        "message": "Server not initialized",
                    },
                }
            result = {"tools": self._registry.list_schemas()}
        elif method == "tools/call":
            # Check permissions for write operations
            if auth_data and self.auth:
                tool_name = params.get("name", "")
                if (
                    tool_name in self._registry.write_tool_names()
                    and not self.auth.check_permission(auth_data, "write")
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
                    "error": {
                        "code": ErrorCodes.SERVER_NOT_INITIALIZED,
                        "message": "Server not initialized",
                    },
                }
            result = await self._call_tool(params)
            # If _call_tool returned an error dict, promote it to a JSON-RPC error
            if (
                isinstance(result, dict)
                and "error" in result
                and "content" not in result
            ):
                return {"jsonrpc": "2.0", "id": req_id, "error": result["error"]}
        elif method == "resources/list":
            resources = []
            for graph_id in graphs:
                graph = graphs[graph_id]
                resources.append(
                    {
                        "uri": f"graph://{graph_id}",
                        "name": graph_id,
                        "description": (
                            f"Graph with {graph.number_of_nodes()} nodes "
                            f"and {graph.number_of_edges()} edges"
                        ),
                        "mimeType": "application/json",
                    }
                )
            result = {"resources": resources}
        elif method == "resources/read":
            uri = params.get("uri", "")
            if not uri.startswith("graph://"):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": f"Invalid resource URI: {uri}",
                    },
                }
            graph_id = uri[len("graph://") :]
            if graph_id not in graphs:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": f"Graph '{graph_id}' not found",
                    },
                }
            graph = graphs[graph_id]
            data = nx.node_link_data(graph, edges="links")
            result = {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(data),
                    }
                ]
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

    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via registry lookup."""
        tool_name: str = params.get("name", "")
        args = params.get("arguments", {})
        logger.debug(f"Executing tool: {tool_name}")

        try:
            # Look up tool in registry
            tool_def = self._registry.get(tool_name)
            if tool_def is None:
                return {
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

            # Validate graph ID if this tool accepts a graph parameter
            if tool_def.graph_param:
                graph_id = args.get(tool_def.graph_param)
                if graph_id is not None:
                    validate_graph_id(graph_id)

            # Call the handler (sync or async)
            if inspect.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(args)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor, tool_def.handler, args
                )

            logger.debug(f"Tool {tool_name} completed successfully")
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        except MCPError as e:
            # Return proper JSON-RPC error for MCP-specific errors
            return {"error": e.to_dict()}

        except nx.NetworkXException as e:
            # NetworkX errors (NetworkXError + algorithm errors like NetworkXNoPath)
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

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s, shutting down...", sig.name)
        self.running = False

    async def _shutdown(self) -> None:
        logger.info("Shutting down server...")
        self.graphs.shutdown()
        self._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Server shutdown complete")

    async def run(self) -> None:
        """Main server loop - read stdin, write stdout."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
            except NotImplementedError:
                pass  # Windows
        try:
            await self._run_loop()
        finally:
            await self._shutdown()

    async def _run_loop(self) -> None:
        """Inner loop split out so run() can wrap with try/finally."""
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
            if os.environ.get("NETWORKX_MCP_INSECURE_CONFIRM", "").lower() != "true":
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
    try:
        asyncio.run(server.run())
    finally:
        if hasattr(server, "_executor"):
            server._executor.shutdown(wait=False)
        server.graphs.shutdown()


# Run the server
if __name__ == "__main__":
    main()
