"""Input validation module for preventing security exploits."""

import logging
import re
from typing import Any, Union

# Type aliases for cleaner annotations
EdgeTuple = Union[
    tuple[Union[str, int], Union[str, int]],
    tuple[Union[str, int], Union[str, int], dict[str, Any]],
]

logger = logging.getLogger(__name__)

# Security constants
MAX_ID_LENGTH = 100
MAX_NODES_PER_REQUEST = 1000
MAX_EDGES_PER_REQUEST = 10000
MAX_STRING_LENGTH = 1000
MAX_ATTRIBUTE_SIZE = 10000

# Regex pattern for safe IDs (alphanumeric, underscore, hyphen)
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")

# Patterns that indicate potential path traversal or injection attacks
DANGEROUS_PATTERNS = [
    re.compile(r"\.\."),  # Path traversal
    re.compile(r'[<>"\']'),  # HTML/SQL injection characters
    re.compile(r";.*--"),  # SQL comment injection
    re.compile(r"(DROP|DELETE|INSERT|UPDATE|SELECT)\s+", re.IGNORECASE),  # SQL keywords
    re.compile(r"[\\\/]etc[\\\/]"),  # System file access
    re.compile(r"[\\\/]proc[\\\/]"),  # Process information
    re.compile(r"\x00"),  # Null bytes
    re.compile(r"[\r\n]"),  # CRLF injection
]


class ValidationError(ValueError):
    """Custom exception for validation errors."""

    pass


def validate_id(value: Any, field_name: str = "ID") -> str:
    """
    Validate an ID field (graph_id, node_id, etc.).

    Args:
        value: The ID value to validate
        field_name: Name of the field for error messages

    Returns:
        Validated string ID

    Raises:
        ValidationError: If validation fails
    """
    # Convert to string and check type
    if value is None:
        raise ValidationError(f"{field_name} cannot be None")

    # Allow integers as IDs
    if isinstance(value, int):
        if value < 0 or value > 999999999:  # Reasonable integer limits
            raise ValidationError(
                f"{field_name} integer must be between 0 and 999999999"
            )
        return str(value)

    # For strings, apply strict validation
    if not isinstance(value, str):
        try:
            str_value = str(value)
        except Exception:
            raise ValidationError(f"{field_name} must be a string or integer")
    else:
        str_value = value

    # Check length
    if len(str_value) == 0:
        raise ValidationError(f"{field_name} cannot be empty")
    if len(str_value) > MAX_ID_LENGTH:
        raise ValidationError(f"{field_name} exceeds maximum length of {MAX_ID_LENGTH}")

    # Check against safe pattern
    if not SAFE_ID_PATTERN.match(str_value):
        raise ValidationError(
            f"{field_name} contains invalid characters. "
            f"Only alphanumeric, underscore, and hyphen allowed (max {MAX_ID_LENGTH} chars)"
        )

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(str_value):
            raise ValidationError(
                f"{field_name} contains potentially dangerous content"
            )

    return str_value


def validate_node_list(nodes: list[Any], max_nodes: int = MAX_NODES_PER_REQUEST) -> list[Union[str, int]]:
    """
    Validate a list[Any] of nodes.

    Args:
        nodes: List of node IDs
        max_nodes: Maximum allowed nodes

    Returns:
        Validated list[Any] of nodes

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(nodes, list[Any]):
        raise ValidationError("Nodes must be provided as a list[Any]")

    if len(nodes) == 0:
        raise ValidationError("Node list[Any] cannot be empty")

    if len(nodes) > max_nodes:
        raise ValidationError(f"Too many nodes. Maximum allowed: {max_nodes}")

    validated_nodes: list[Union[str, int]] = []
    seen = set[Any]()

    for i, node in enumerate(nodes):
        # Allow integers directly
        if isinstance(node, int):
            if node < 0 or node > 999999999:
                raise ValidationError(
                    f"Node at index {i}: integer must be between 0 and 999999999"
                )
            validated_node: Union[str, int] = node
        else:
            # Validate as string ID
            try:
                validated_node = validate_id(node, f"Node at index {i}")
            except ValidationError:
                # If it's a tuple[Any, ...] (for nodes with attributes), validate only the ID part
                if isinstance(node, (list[Any], tuple[Any, ...])) and len(node) >= 1:
                    validated_node = validate_id(node[0], f"Node ID at index {i}")
                    # Note: We only store the ID, not the full tuple[Any, ...] for this return type
                else:
                    raise

        # Check for duplicates
        node_key = (
            validated_node
            if not isinstance(validated_node, tuple[Any, ...])
            else validated_node[0]
        )
        if node_key in seen:
            raise ValidationError(f"Duplicate node ID: {node_key}")
        seen.add(node_key)

        validated_nodes.append(validated_node)

    return validated_nodes


def validate_edge_list(edges: list[Any], max_edges: int = MAX_EDGES_PER_REQUEST) -> list[EdgeTuple]:
    """
    Validate a list[Any] of edges.

    Args:
        edges: List of edges (each edge is a tuple[Any, ...]/list[Any] of 2-3 elements)
        max_edges: Maximum allowed edges

    Returns:
        Validated list[Any] of edge tuples

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(edges, list[Any]):
        raise ValidationError("Edges must be provided as a list[Any]")

    if len(edges) == 0:
        raise ValidationError("Edge list[Any] cannot be empty")

    if len(edges) > max_edges:
        raise ValidationError(f"Too many edges. Maximum allowed: {max_edges}")

    validated_edges: list[EdgeTuple] = []

    for i, edge in enumerate(edges):
        if not isinstance(edge, (list[Any], tuple[Any, ...])):
            raise ValidationError(f"Edge at index {i} must be a list[Any] or tuple[Any, ...]")

        if len(edge) < 2:
            raise ValidationError(
                f"Edge at index {i} must have at least 2 elements (source, target)"
            )

        if len(edge) > 3:
            raise ValidationError(
                f"Edge at index {i} has too many elements (max 3: source, target, attributes)"
            )

        # Validate source and target
        if isinstance(edge[0], int) and isinstance(edge[1], int):
            # Allow integer node IDs
            if edge[0] < 0 or edge[0] > 999999999 or edge[1] < 0 or edge[1] > 999999999:
                raise ValidationError(
                    f"Edge at index {i}: node IDs must be between 0 and 999999999"
                )
            source: Union[str, int] = edge[0]
            target: Union[str, int] = edge[1]
        else:
            source = validate_id(edge[0], f"Source node in edge {i}")
            target = validate_id(edge[1], f"Target node in edge {i}")

        # If there are attributes (3rd element), validate them
        if len(edge) == 3:
            attrs = validate_attributes(edge[2], f"Edge {i} attributes")
            validated_edges.append((source, target, attrs))
        else:
            validated_edges.append((source, target))

    return validated_edges


def validate_attributes(attrs: Any, field_name: str = "Attributes") -> dict[str, Any]:
    """
    Validate node or edge attributes.

    Args:
        attrs: Attributes dictionary
        field_name: Name of the field for error messages

    Returns:
        Validated attributes dictionary

    Raises:
        ValidationError: If validation fails
    """
    if attrs is None:
        return {}

    if not isinstance(attrs, dict[str, Any]):
        raise ValidationError(f"{field_name} must be a dictionary")

    # Check size
    if len(str(attrs)) > MAX_ATTRIBUTE_SIZE:
        raise ValidationError(f"{field_name} exceeds maximum size")

    validated_attrs = {}

    for key, value in attrs.items():
        # Validate attribute key
        if not isinstance(key, str):
            raise ValidationError(f"{field_name}: keys must be strings")

        if len(key) > MAX_ID_LENGTH:
            raise ValidationError(f"{field_name}: key '{key}' exceeds maximum length")

        # Check for dangerous patterns in key
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(key):
                raise ValidationError(
                    f"{field_name}: key '{key}' contains potentially dangerous content"
                )

        # Validate attribute value
        validated_value = sanitize_value(value, f"{field_name}['{key}']")
        validated_attrs[key] = validated_value

    return validated_attrs


def sanitize_value(value: Any, field_name: str = "Value") -> Any:
    """
    Sanitize a single value to prevent injection attacks.

    Args:
        value: Value to sanitize
        field_name: Name of the field for error messages

    Returns:
        Sanitized value

    Raises:
        ValidationError: If value is dangerous
    """
    # Allow basic types
    if value is None or isinstance(value, (bool, int, float)):
        # Check numeric bounds
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if abs(value) > 1e9:
                raise ValidationError(f"{field_name}: numeric value too large")
        return value

    # Sanitize strings
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ValidationError(
                f"{field_name}: string exceeds maximum length of {MAX_STRING_LENGTH}"
            )

        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(value):
                # Log potential attack attempt
                logger.warning(
                    f"Potential injection attempt in {field_name}: pattern matched"
                )
                raise ValidationError(
                    f"{field_name} contains potentially dangerous content"
                )

        return value

    # Handle lists and tuples
    if isinstance(value, (list[Any], tuple[Any, ...])):
        if len(value) > 100:  # Reasonable limit for attribute lists
            raise ValidationError(f"{field_name}: list[Any] too long (max 100 elements)")

        sanitized_list = []
        for i, item in enumerate(value):
            sanitized_list.append(sanitize_value(item, f"{field_name}[{i}]"))
        return sanitized_list if isinstance(value, list[Any]) else tuple[Any, ...](sanitized_list)

    # Handle nested dictionaries (with depth limit)
    if isinstance(value, dict[str, Any]):
        if len(value) > 50:  # Reasonable limit for nested dicts
            raise ValidationError(f"{field_name}: dictionary too large (max 50 keys)")

        sanitized_dict: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValidationError(f"{field_name}: dictionary keys must be strings")
            sanitized_dict[k] = sanitize_value(v, f"{field_name}['{k}']")
        return sanitized_dict

    # Reject other types
    raise ValidationError(f"{field_name}: unsupported type {type(value).__name__}")


def validate_graph_type(graph_type: str) -> str:
    """
    Validate graph type parameter.

    Args:
        graph_type: Type of graph

    Returns:
        Validated graph type

    Raises:
        ValidationError: If invalid type
    """
    valid_types = {"undirected", "directed", "multi", "multi_directed"}

    if not isinstance(graph_type, str):
        raise ValidationError("Graph type must be a string")

    graph_type = graph_type.lower().strip()

    if graph_type not in valid_types:
        raise ValidationError(
            f"Invalid graph type. Must be one of: {', '.join(valid_types)}"
        )

    return graph_type


def safe_error_message(error: Exception) -> str:
    """
    Create a safe error message without exposing stack traces or sensitive info.

    Args:
        error: The exception

    Returns:
        Safe error message
    """
    if isinstance(error, ValidationError):
        return str(error)

    # For other exceptions, return generic message
    error_type = type(error).__name__

    # Map common errors to safe messages
    safe_messages = {
        "KeyError": "Requested resource not found",
        "ValueError": "Invalid input provided",
        "TypeError": "Invalid data type provided",
        "AttributeError": "Invalid operation requested",
        "NetworkXError": "Graph operation failed",
        "NetworkXNoPath": "No path exists between the specified nodes",
    }

    return safe_messages.get(error_type, "Operation failed. Please check your input.")
