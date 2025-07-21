"""Input validation and sanitization for security.

This module provides comprehensive input validation and sanitization
for graph operations and user data.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    sanitized_data: Any | None = None
    errors: list[str] = field(default_factory=list[Any])


class RequestValidator:
    """Validates incoming MCP requests."""

    @staticmethod
    def validate_graph_id(graph_id: str) -> ValidationResult:
        """Validate graph ID format."""
        if not graph_id or not isinstance(graph_id, str):
            return ValidationResult(
                False, None, ["Graph ID must be a non-empty string"]
            )

        # Basic alphanumeric + underscore/hyphen check
        if not re.match(r"^[a-zA-Z0-9_-]+$", graph_id):
            return ValidationResult(
                False,
                None,
                [
                    "Graph ID can only contain letters, numbers, underscores, and hyphens"
                ],
            )

        # Length check
        if len(graph_id) > 100:
            return ValidationResult(
                False, None, ["Graph ID must be 100 characters or less"]
            )

        return ValidationResult(True, graph_id.strip())

    @staticmethod
    def validate_node_id(node_id: Any) -> ValidationResult:
        """Validate node ID."""
        if node_id is None:
            return ValidationResult(False, None, ["Node ID cannot be None"])

        # Allow string, int, or float
        if isinstance(node_id, (str, int, float)):
            # For strings, apply similar rules as graph_id but allow more characters
            if isinstance(node_id, str):
                if len(str(node_id)) > 200:
                    return ValidationResult(
                        False, None, ["Node ID must be 200 characters or less"]
                    )
            return ValidationResult(True, node_id)

        return ValidationResult(False, None, ["Node ID must be string, int, or float"])

    @staticmethod
    def validate_node_data(data: dict[str, Any]) -> ValidationResult:
        """Sanitize and validate node attributes."""
        if not isinstance(data, dict[str, Any]):
            return ValidationResult(False, None, ["Node data must be a dictionary"])

        # Remove potentially dangerous keys
        dangerous_patterns = [
            "__",  # Python special attributes
            "eval",  # Code execution functions
            "exec",
            "compile",
            "import",
            "open",
            "file",
            "input",
            "raw_input",
        ]

        clean_data = {}
        errors = []

        for key, value in data.items():
            # Check key safety
            if not isinstance(key, str):
                errors.append(f"Attribute key must be string, got {type(key)}")
                continue

            # Check for dangerous patterns in key
            if any(pattern in key.lower() for pattern in dangerous_patterns):
                errors.append(
                    f"Attribute key '{key}' contains potentially dangerous pattern"
                )
                continue

            # Validate value
            if RequestValidator._is_safe_value(value):
                clean_data[key] = value
            else:
                errors.append(f"Attribute value for '{key}' is not safe")

        if errors:
            return ValidationResult(False, clean_data, errors)

        return ValidationResult(True, clean_data)

    @staticmethod
    def validate_edge_data(edge: list[Any] | tuple[Any, ...]) -> ValidationResult:
        """Validate edge format."""
        if not isinstance(edge, (list[Any], tuple[Any, ...])):
            return ValidationResult(False, None, ["Edge must be a list[Any] or tuple[Any, ...]"])

        if len(edge) < 2:
            return ValidationResult(False, None, ["Edge must have at least 2 nodes"])

        if len(edge) > 3:
            return ValidationResult(
                False, None, ["Edge can have at most 3 elements (source, target, data)"]
            )

        # Validate source and target nodes
        source_result = RequestValidator.validate_node_id(edge[0])
        if not source_result.is_valid:
            return ValidationResult(
                False, None, [f"Source node: {source_result.errors[0]}"]
            )

        target_result = RequestValidator.validate_node_id(edge[1])
        if not target_result.is_valid:
            return ValidationResult(
                False, None, [f"Target node: {target_result.errors[0]}"]
            )

        # Validate edge data if present
        clean_edge = [source_result.sanitized_data, target_result.sanitized_data]

        if len(edge) == 3:
            if isinstance(edge[2], dict[str, Any]):
                data_result = RequestValidator.validate_node_data(edge[2])
                if not data_result.is_valid:
                    return ValidationResult(
                        False, None, [f"Edge data: {', '.join(data_result.errors)}"]
                    )
                clean_edge.append(data_result.sanitized_data)
            else:
                return ValidationResult(False, None, ["Edge data must be a dictionary"])

        return ValidationResult(True, clean_edge)

    @staticmethod
    def _is_safe_value(value: Any) -> bool:
        """Check if a value is safe to store."""
        # Allow basic types
        if isinstance(value, (str, int, float, bool, type(None))):
            return True

        # Allow lists of safe values
        if isinstance(value, list[Any]):
            return all(RequestValidator._is_safe_value(item) for item in value)

        # Allow dicts of safe values
        if isinstance(value, dict[str, Any]):
            return all(
                isinstance(k, str) and RequestValidator._is_safe_value(v)
                for k, v in value.items()
            )

        # Reject other types (functions, classes, etc.)
        return False


class SecurityValidator:
    """Additional security validation for operations."""

    @staticmethod
    def validate_algorithm_parameters(algorithm: str, params: dict[str, Any]) -> ValidationResult:
        """Validate algorithm parameters for safety."""
        if not isinstance(algorithm, str):
            return ValidationResult(False, None, ["Algorithm name must be a string"])

        # Whitelist of allowed algorithms
        allowed_algorithms = {
            "shortest_path",
            "all_shortest_paths",
            "dijkstra_path",
            "betweenness_centrality",
            "closeness_centrality",
            "degree_centrality",
            "pagerank",
            "connected_components",
            "strongly_connected_components",
            "clustering",
            "triangles",
            "diameter",
            "density",
            "is_connected",
            "minimum_spanning_tree",
            "maximum_flow",
            "community_detection",
        }

        if algorithm not in allowed_algorithms:
            return ValidationResult(
                False, None, [f"Algorithm '{algorithm}' is not allowed"]
            )

        # Validate parameters
        if not isinstance(params, dict[str, Any]):
            return ValidationResult(False, None, ["Parameters must be a dictionary"])

        # Basic parameter validation
        clean_params = {}
        errors = []

        for key, value in params.items():
            if not isinstance(key, str):
                errors.append(f"Parameter key must be string, got {type(key)}")
                continue

            # Basic safety checks for parameter values
            if RequestValidator._is_safe_value(value):
                clean_params[key] = value
            else:
                errors.append(f"Parameter '{key}' has unsafe value")

        if errors:
            return ValidationResult(False, clean_params, errors)

        return ValidationResult(True, clean_params)

    @staticmethod
    def validate_file_path(file_path: str) -> ValidationResult:
        """Validate file path for import/export operations."""
        if not isinstance(file_path, str):
            return ValidationResult(False, None, ["File path must be a string"])

        # Basic path traversal protection
        if ".." in file_path or file_path.startswith("/"):
            return ValidationResult(
                False, None, ["File path cannot contain '..' or start with '/'"]
            )

        # Allow only certain file extensions
        allowed_extensions = {".json", ".csv", ".txt", ".graphml", ".gexf"}
        if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
            return ValidationResult(False, None, ["File extension not allowed"])

        return ValidationResult(True, file_path.strip())

    @staticmethod
    def validate_resource_limits(operation: str, **limits: Any) -> ValidationResult:
        """Validate resource usage limits."""
        errors = []

        # Check node count limits
        if "node_count" in limits:
            node_count = limits["node_count"]
            if not isinstance(node_count, int) or node_count < 0:
                errors.append("Node count must be a non-negative integer")
            elif node_count > 100000:  # Arbitrary limit
                errors.append("Node count exceeds maximum limit (100,000)")

        # Check edge count limits
        if "edge_count" in limits:
            edge_count = limits["edge_count"]
            if not isinstance(edge_count, int) or edge_count < 0:
                errors.append("Edge count must be a non-negative integer")
            elif edge_count > 1000000:  # Arbitrary limit
                errors.append("Edge count exceeds maximum limit (1,000,000)")

        # Check memory usage
        if "memory_mb" in limits:
            memory_mb = limits["memory_mb"]
            if not isinstance(memory_mb, (int, float)) or memory_mb < 0:
                errors.append("Memory limit must be a non-negative number")
            elif memory_mb > 2048:  # 2GB limit
                errors.append("Memory usage exceeds maximum limit (2GB)")

        if errors:
            return ValidationResult(False, None, errors)

        return ValidationResult(True, limits)
