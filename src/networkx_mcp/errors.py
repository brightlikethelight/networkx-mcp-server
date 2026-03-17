"""Comprehensive error handling for NetworkX MCP Server.

This module provides proper JSON-RPC 2.0 compliant error handling with specific
error codes and meaningful messages for all possible error conditions.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# JSON-RPC 2.0 Error Codes
class ErrorCodes:
    """JSON-RPC 2.0 standard error codes plus MCP-specific codes."""

    # JSON-RPC 2.0 Standard Errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-Specific Server Errors (-32000 to -32099)
    GRAPH_NOT_FOUND = -32001
    NODE_NOT_FOUND = -32002
    EDGE_NOT_FOUND = -32003
    GRAPH_ALREADY_EXISTS = -32004
    INVALID_GRAPH_ID = -32005
    INVALID_NODE_ID = -32006
    INVALID_EDGE = -32007
    GRAPH_OPERATION_FAILED = -32008
    ALGORITHM_ERROR = -32009
    VALIDATION_ERROR = -32010
    RESOURCE_LIMIT_EXCEEDED = -32011
    SERVER_NOT_INITIALIZED = -32012  # MCP-specific


class MCPError(Exception):
    """Base exception for MCP server errors.

    All MCP errors should inherit from this class to ensure consistent
    error handling and JSON-RPC 2.0 compliance.
    """

    def __init__(self, code: int, message: str, data: Optional[Any] = None) -> None:
        """Initialize MCP error.

        Args:
            code: JSON-RPC 2.0 error code
            message: Human-readable error message
            data: Optional additional error data
        """
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to JSON-RPC 2.0 error object."""
        error_dict = {"code": self.code, "message": self.message}
        if self.data is not None:
            error_dict["data"] = self.data
        return error_dict


class GraphNotFoundError(MCPError):
    """Raised when attempting to access a non-existent graph."""

    def __init__(self, graph_id: str) -> None:
        super().__init__(
            ErrorCodes.GRAPH_NOT_FOUND,
            f"Graph '{graph_id}' not found",
            {"graph_id": graph_id},
        )
        self.graph_id = graph_id


class NodeNotFoundError(MCPError):
    """Raised when attempting to access a non-existent node."""

    def __init__(self, graph_id: str, node_id: str) -> None:
        super().__init__(
            ErrorCodes.NODE_NOT_FOUND,
            f"Node '{node_id}' not found in graph '{graph_id}'",
            {"graph_id": graph_id, "node_id": node_id},
        )
        self.graph_id = graph_id
        self.node_id = node_id


class EdgeNotFoundError(MCPError):
    """Raised when attempting to access a non-existent edge."""

    def __init__(self, graph_id: str, source: str, target: str) -> None:
        super().__init__(
            ErrorCodes.EDGE_NOT_FOUND,
            f"Edge '{source}' -> '{target}' not found in graph '{graph_id}'",
            {"graph_id": graph_id, "source": source, "target": target},
        )
        self.graph_id = graph_id
        self.source = source
        self.target = target


class GraphAlreadyExistsError(MCPError):
    """Raised when attempting to create a graph that already exists."""

    def __init__(self, graph_id: str) -> None:
        super().__init__(
            ErrorCodes.GRAPH_ALREADY_EXISTS,
            f"Graph '{graph_id}' already exists",
            {"graph_id": graph_id},
        )
        self.graph_id = graph_id


class InvalidGraphIdError(MCPError):
    """Raised when graph ID is invalid or malformed."""

    def __init__(self, graph_id: str, reason: str = "Invalid format") -> None:
        # Don't include the full malicious input in the error message for security
        safe_message = f"Invalid graph ID: {reason}"
        super().__init__(
            ErrorCodes.INVALID_GRAPH_ID,
            safe_message,
            {"reason": reason},  # Don't include full graph_id in data either
        )
        self.graph_id = graph_id


class InvalidNodeIdError(MCPError):
    """Raised when node ID is invalid or malformed."""

    def __init__(self, node_id: str, reason: str = "Invalid format") -> None:
        # Don't include the full malicious input in the error message for security
        safe_message = f"Invalid node ID: {reason}"
        super().__init__(
            ErrorCodes.INVALID_NODE_ID,
            safe_message,
            {"reason": reason},  # Don't include full node_id in data either
        )
        self.node_id = node_id


class InvalidEdgeError(MCPError):
    """Raised when edge specification is invalid."""

    def __init__(self, edge_data: Any, reason: str = "Invalid edge format") -> None:
        super().__init__(
            ErrorCodes.INVALID_EDGE,
            f"Invalid edge: {reason}",
            {"edge_data": edge_data, "reason": reason},
        )
        self.edge_data = edge_data


class GraphOperationError(MCPError):
    """Raised when a graph operation fails."""

    def __init__(self, operation: str, graph_id: str, reason: str) -> None:
        super().__init__(
            ErrorCodes.GRAPH_OPERATION_FAILED,
            f"Graph operation '{operation}' failed on graph '{graph_id}': {reason}",
            {"operation": operation, "graph_id": graph_id, "reason": reason},
        )
        self.operation = operation
        self.graph_id = graph_id


class AlgorithmError(MCPError):
    """Raised when graph algorithm execution fails."""

    def __init__(self, algorithm: str, graph_id: str, reason: str) -> None:
        super().__init__(
            ErrorCodes.ALGORITHM_ERROR,
            f"Algorithm '{algorithm}' failed on graph '{graph_id}': {reason}",
            {"algorithm": algorithm, "graph_id": graph_id, "reason": reason},
        )
        self.algorithm = algorithm
        self.graph_id = graph_id


class ValidationError(MCPError):
    """Raised when input validation fails."""

    def __init__(self, parameter: str, value: Any, reason: str) -> None:
        super().__init__(
            ErrorCodes.VALIDATION_ERROR,
            f"Validation failed for parameter '{parameter}': {reason}",
            {"parameter": parameter, "value": value, "reason": reason},
        )
        self.parameter = parameter
        self.value = value


class ResourceLimitExceededError(MCPError):
    """Raised when resource limits are exceeded."""

    def __init__(self, resource: str, limit: Any, current: Any) -> None:
        super().__init__(
            ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
            f"Resource limit exceeded for {resource}: {current} > {limit}",
            {"resource": resource, "limit": limit, "current": current},
        )
        self.resource = resource
        self.limit = limit
        self.current = current


class ServerNotInitializedError(MCPError):
    """Raised when MCP server is not properly initialized."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            ErrorCodes.SERVER_NOT_INITIALIZED,
            f"Server not initialized - cannot perform '{operation}'",
            {"operation": operation},
        )
        self.operation = operation


def validate_graph_id(graph_id: Any) -> str:
    """Validate and normalize graph ID.

    Args:
        graph_id: Graph ID to validate

    Returns:
        Validated graph ID string

    Raises:
        InvalidGraphIdError: If graph ID is invalid
    """
    if graph_id is None:
        raise InvalidGraphIdError("None", "Graph ID cannot be None")

    if not isinstance(graph_id, str):
        raise InvalidGraphIdError(str(graph_id), "Graph ID must be a string")

    if not graph_id.strip():
        raise InvalidGraphIdError(graph_id, "Graph ID cannot be empty")

    graph_id = graph_id.strip()

    if len(graph_id) > 100:
        raise InvalidGraphIdError(graph_id, "Graph ID too long (max 100 characters)")

    # Only allow alphanumeric, underscore, and hyphen
    if not all(c.isalnum() or c in "_-" for c in graph_id):
        raise InvalidGraphIdError(
            graph_id,
            "Graph ID can only contain letters, numbers, underscore, and hyphen",
        )

    # Security check - prevent path traversal
    if ".." in graph_id or "/" in graph_id or "\\" in graph_id:
        raise InvalidGraphIdError(graph_id, "Graph ID cannot contain path elements")

    return str(graph_id)
