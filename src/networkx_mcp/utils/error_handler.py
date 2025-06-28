"""Enhanced error handling and user-friendly error messages for NetworkX MCP Server."""

import logging
import traceback

from typing import Any
from typing import Dict
from typing import List
from typing import Optional


logger = logging.getLogger(__name__)


class NetworkXMCPError(Exception):
    """Base exception for NetworkX MCP Server."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class GraphNotFoundError(NetworkXMCPError):
    """Raised when a requested graph is not found."""

    def __init__(self, graph_id: str, available_graphs: List[str]):
        available_list = ", ".join(available_graphs[:5])
        if len(available_graphs) > 5:
            available_list += f" and {len(available_graphs) - 5} more"

        message = (
            f"Graph '{graph_id}' not found. "
            f"Available graphs: {available_list if available_graphs else 'none'}"
        )

        super().__init__(message, {
            "graph_id": graph_id,
            "available_graphs": available_graphs,
            "suggestion": "Use list_graphs() to see all available graphs"
        })


class InvalidGraphTypeError(NetworkXMCPError):
    """Raised when an invalid graph type is specified."""

    def __init__(self, provided_type: str, valid_types: List[str]):
        message = (
            f"Invalid graph type '{provided_type}'. "
            f"Valid types: {', '.join(valid_types)}. "
            f"Example: create_graph('my_graph', graph_type='Graph')"
        )

        super().__init__(message, {
            "provided_type": provided_type,
            "valid_types": valid_types,
            "suggestion": "Use 'Graph' for undirected, 'DiGraph' for directed graphs"
        })


class NodeNotFoundError(NetworkXMCPError):
    """Raised when a requested node is not found in the graph."""

    def __init__(self, node_id: Any, graph_id: str, available_nodes: List[Any]):
        node_list = ", ".join(str(n) for n in available_nodes[:10])
        if len(available_nodes) > 10:
            node_list += f" and {len(available_nodes) - 10} more"

        message = (
            f"Node '{node_id}' not found in graph '{graph_id}'. "
            f"Available nodes: {node_list if available_nodes else 'none'}"
        )

        super().__init__(message, {
            "node_id": node_id,
            "graph_id": graph_id,
            "available_nodes": available_nodes[:100],  # Limit for serialization
            "suggestion": "Check node ID spelling or add the node first"
        })


class EdgeNotFoundError(NetworkXMCPError):
    """Raised when a requested edge is not found in the graph."""

    def __init__(self, source: Any, target: Any, graph_id: str):
        message = (
            f"Edge ({source}, {target}) not found in graph '{graph_id}'. "
            f"Make sure both nodes exist and are connected."
        )

        super().__init__(message, {
            "source": source,
            "target": target,
            "graph_id": graph_id,
            "suggestion": "Check if nodes exist and add edge if needed"
        })


class InvalidParameterError(NetworkXMCPError):
    """Raised when invalid parameters are provided to functions."""

    def __init__(self, parameter: str, value: Any, expected_type: str, valid_values: Optional[List[Any]] = None):
        if valid_values:
            message = (
                f"Invalid value '{value}' for parameter '{parameter}'. "
                f"Expected one of: {', '.join(str(v) for v in valid_values)}"
            )
        else:
            message = (
                f"Invalid value '{value}' for parameter '{parameter}'. "
                f"Expected type: {expected_type}"
            )

        super().__init__(message, {
            "parameter": parameter,
            "provided_value": value,
            "expected_type": expected_type,
            "valid_values": valid_values,
            "suggestion": f"Provide a valid {expected_type} value"
        })


class AlgorithmError(NetworkXMCPError):
    """Raised when graph algorithms encounter errors."""

    def __init__(self, algorithm: str, graph_id: str, reason: str, suggestions: Optional[List[str]] = None):
        message = f"Algorithm '{algorithm}' failed on graph '{graph_id}': {reason}"

        super().__init__(message, {
            "algorithm": algorithm,
            "graph_id": graph_id,
            "reason": reason,
            "suggestions": suggestions or []
        })


class FileOperationError(NetworkXMCPError):
    """Raised when file I/O operations fail."""

    def __init__(self, operation: str, file_path: str, reason: str):
        message = f"File {operation} failed for '{file_path}': {reason}"

        super().__init__(message, {
            "operation": operation,
            "file_path": file_path,
            "reason": reason,
            "suggestion": "Check file path and permissions"
        })


class ErrorHandler:
    """Centralized error handling with user-friendly messages and debugging support."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.error_counts: Dict[str, int] = {}

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle an error and return a user-friendly error response.

        Args:
            error: The exception that occurred
            context: Additional context about the operation
            operation: Name of the operation that failed

        Returns:
            Dictionary with error information and suggestions
        """
        context = context or {}
        error_type = type(error).__name__

        # Count error occurrences
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error for debugging
        logger.error(f"Error in {operation or 'operation'}: {error}", exc_info=self.debug_mode)

        # Create user-friendly error response
        error_response = {
            "error": True,
            "error_type": error_type,
            "message": str(error),
            "operation": operation,
            "timestamp": self._get_timestamp()
        }

        # Add context if available
        if context:
            error_response["context"] = context

        # Add debug information if in debug mode
        if self.debug_mode:
            error_response["debug"] = {
                "traceback": traceback.format_exc(),
                "error_counts": self.error_counts.copy()
            }

        # Add specific handling for known error types
        if isinstance(error, NetworkXMCPError):
            error_response["details"] = error.details
        elif isinstance(error, ValueError):
            error_response["suggestions"] = self._get_value_error_suggestions(str(error))
        elif isinstance(error, KeyError):
            error_response["suggestions"] = self._get_key_error_suggestions(str(error))
        elif isinstance(error, FileNotFoundError):
            error_response["suggestions"] = ["Check file path", "Ensure file exists", "Verify file permissions"]
        elif isinstance(error, ImportError):
            error_response["suggestions"] = self._get_import_error_suggestions(str(error))

        return error_response

    def validate_graph_id(self, graph_id: Any, available_graphs: List[str]) -> str:
        """Validate and normalize graph ID."""
        if not isinstance(graph_id, str):
            msg = "graph_id"
            raise InvalidParameterError(
                msg, graph_id, "string",
                ["string identifier like 'my_graph'"]
            )

        if not graph_id.strip():
            msg = "graph_id"
            raise InvalidParameterError(
                msg, graph_id, "non-empty string",
                ["'social_network'", "'transport_graph'", "'citation_net'"]
            )

        if graph_id not in available_graphs:
            raise GraphNotFoundError(graph_id, available_graphs)

        return graph_id

    def validate_node_exists(self, node_id: Any, graph_id: str, graph) -> Any:
        """Validate that a node exists in the graph."""
        if node_id not in graph.nodes():
            available_nodes = list(graph.nodes())
            raise NodeNotFoundError(node_id, graph_id, available_nodes)
        return node_id

    def validate_edge_exists(self, source: Any, target: Any, graph_id: str, graph) -> tuple:
        """Validate that an edge exists in the graph."""
        if not graph.has_edge(source, target):
            raise EdgeNotFoundError(source, target, graph_id)
        return (source, target)

    def _get_value_error_suggestions(self, error_message: str) -> List[str]:
        """Get suggestions for ValueError."""
        suggestions = []
        message_lower = error_message.lower()

        if "graph type" in message_lower:
            suggestions.append("Use 'Graph', 'DiGraph', 'MultiGraph', or 'MultiDiGraph'")
        elif "algorithm" in message_lower:
            suggestions.append("Check algorithm name spelling")
            suggestions.append("See documentation for supported algorithms")
        elif "node" in message_lower:
            suggestions.append("Verify node exists in graph")
            suggestions.append("Check node ID format")
        elif "weight" in message_lower:
            suggestions.append("Ensure edge has weight attribute")
            suggestions.append("Use unweighted algorithms for unweighted graphs")

        return suggestions

    def _get_key_error_suggestions(self, error_message: str) -> List[str]:
        """Get suggestions for KeyError."""
        return [
            "Check if the key exists",
            "Verify spelling of attribute/parameter names",
            "Use list_graphs() to see available graphs"
        ]

    def _get_import_error_suggestions(self, error_message: str) -> List[str]:
        """Get suggestions for ImportError."""
        suggestions = ["Install missing dependencies"]

        if "plotly" in error_message.lower():
            suggestions.append("pip install plotly")
        elif "matplotlib" in error_message.lower():
            suggestions.append("pip install matplotlib")
        elif "community" in error_message.lower():
            suggestions.append("pip install python-louvain")

        return suggestions

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        from datetime import timezone
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        total_errors = sum(self.error_counts.values())
        return {
            "total_errors": total_errors,
            "error_types": self.error_counts.copy(),
            "most_common": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_mcp_error(operation: str):
    """Decorator for MCP tool functions to handle errors gracefully."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_response = error_handler.handle_error(
                    e,
                    context={"args": args, "kwargs": kwargs},
                    operation=operation
                )
                # Return error response instead of raising
                return error_response
        return wrapper
    return decorator


def set_debug_mode(enabled: bool):
    """Enable or disable debug mode for detailed error information."""
    error_handler.debug_mode = enabled


def get_error_stats() -> Dict[str, Any]:
    """Get error statistics for monitoring."""
    return error_handler.get_error_statistics()
