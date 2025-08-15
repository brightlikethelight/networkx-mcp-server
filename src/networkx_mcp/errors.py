"""Comprehensive error handling for NetworkX MCP Server.

This module provides proper JSON-RPC 2.0 compliant error handling with specific
error codes and meaningful messages for all possible error conditions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

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
    SERVER_NOT_INITIALIZED = -32002  # MCP-specific


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


def handle_error(error: Exception, operation: str = "unknown") -> Dict[str, Any]:
    """Convert any exception to proper JSON-RPC error response.

    Args:
        error: The exception that occurred
        operation: The operation that failed (for logging)

    Returns:
        JSON-RPC 2.0 error object
    """
    if isinstance(error, MCPError):
        # Already a proper MCP error
        logger.error(f"MCP error in {operation}: {error.message}")
        return error.to_dict()

    # Handle NetworkX-specific exceptions
    if hasattr(error, "__module__") and "networkx" in error.__module__:
        logger.error(f"NetworkX error in {operation}: {str(error)}")

        # Check if it's an algorithm-specific error
        error_msg = str(error).lower()
        if (
            "no path" in error_msg
            or "not connected" in error_msg
            or "no route" in error_msg
            or "unreachable" in error_msg
        ):
            return MCPError(
                ErrorCodes.ALGORITHM_ERROR,
                f"Algorithm error: {str(error)}",
                {"operation": operation, "algorithm_error": str(error)},
            ).to_dict()
        else:
            return MCPError(
                ErrorCodes.GRAPH_OPERATION_FAILED,
                f"Graph operation failed: {str(error)}",
                {"operation": operation, "networkx_error": str(error)},
            ).to_dict()

    # Handle generic exceptions
    logger.exception(f"Unexpected error in {operation}")
    return MCPError(
        ErrorCodes.INTERNAL_ERROR,
        "Internal server error",
        {"operation": operation, "error_type": type(error).__name__},
    ).to_dict()


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


def validate_node_id(node_id: Any) -> str:
    """Validate and normalize node ID.

    Args:
        node_id: Node ID to validate

    Returns:
        Validated node ID string

    Raises:
        InvalidNodeIdError: If node ID is invalid
    """
    if node_id is None:
        raise InvalidNodeIdError("None", "Node ID cannot be None")

    if not isinstance(node_id, (str, int)):
        raise InvalidNodeIdError(str(node_id), "Node ID must be a string or integer")

    node_id_str = str(node_id)

    if not node_id_str.strip():
        raise InvalidNodeIdError(node_id_str, "Node ID cannot be empty")

    node_id_str = node_id_str.strip()

    if len(node_id_str) > 100:
        raise InvalidNodeIdError(node_id_str, "Node ID too long (max 100 characters)")

    return node_id_str


def validate_edge(edge: Any) -> Tuple[str, str]:
    """Validate edge specification.

    Args:
        edge: Edge specification (should be [source, target])

    Returns:
        Tuple of (source, target) as strings

    Raises:
        InvalidEdgeError: If edge specification is invalid
    """
    if not isinstance(edge, (list, tuple)):
        raise InvalidEdgeError(edge, "Edge must be a list or tuple")

    if len(edge) != 2:
        raise InvalidEdgeError(
            edge, "Edge must have exactly 2 elements [source, target]"
        )

    try:
        source = validate_node_id(edge[0])
        target = validate_node_id(edge[1])
    except InvalidNodeIdError as e:
        raise InvalidEdgeError(edge, f"Invalid node in edge: {e.message}")

    return source, target


def validate_required_params(params: Dict[str, Any], required: List[str]) -> None:
    """Validate that all required parameters are present.

    Args:
        params: Parameter dictionary
        required: List of required parameter names

    Raises:
        ValidationError: If any required parameter is missing
    """
    for param in required:
        if param not in params:
            raise ValidationError(param, None, "Required parameter missing")

        if params[param] is None:
            raise ValidationError(param, None, "Required parameter cannot be None")


def validate_centrality_measures(measures: Any) -> List[str]:
    """Validate centrality measures list.

    Args:
        measures: List of centrality measures

    Returns:
        Validated list of centrality measures

    Raises:
        ValidationError: If measures are invalid
    """
    if not isinstance(measures, list):
        raise ValidationError("measures", measures, "Measures must be a list")

    if not measures:
        raise ValidationError("measures", measures, "Measures list cannot be empty")

    valid_measures = {"degree", "betweenness", "closeness", "eigenvector"}

    for measure in measures:
        if not isinstance(measure, str):
            raise ValidationError(
                "measures", measures, f"Measure must be string, got {type(measure)}"
            )

        if measure not in valid_measures:
            raise ValidationError(
                "measures",
                measures,
                f"Invalid measure '{measure}'. Valid measures: {valid_measures}",
            )

    return measures


if __name__ == "__main__":
    # Test error handling
    try:
        validate_graph_id("../../../etc/passwd")
    except InvalidGraphIdError as e:
        print(f"✅ Security validation works: {e.message}")

    try:
        validate_edge(["valid", "valid"])
        print("✅ Valid edge passes validation")
    except InvalidEdgeError as e:
        print(f"❌ Unexpected error: {e.message}")

    try:
        validate_edge("invalid")
    except InvalidEdgeError as e:
        print(f"✅ Invalid edge rejected: {e.message}")

    print("✅ Error handling module working correctly")
