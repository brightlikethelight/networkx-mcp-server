"""Comprehensive input validation and sanitization."""

import html
import re

from datetime import datetime
from typing import Any
from typing import Dict
from typing import Union


class SecurityError(Exception):
    """Base exception for security violations."""
    pass


class ValidationError(SecurityError):
    """Raised when input validation fails."""
    pass


class SecurityValidator:
    """Comprehensive input validation and sanitization."""

    # Allowed patterns - strict for security
    GRAPH_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$")
    USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,63}$")
    NODE_ID_PATTERN = re.compile(r'^[^<>&"\']{1,1000}$')

    # Control characters that should never appear in user input
    CONTROL_CHARS = re.compile(r"[\x00-\x1F\x7F-\x9F]")

    # Size limits to prevent DoS
    MAX_STRING_LENGTH = 10_000
    MAX_IDENTIFIER_LENGTH = 100
    MAX_DICT_SIZE = 1_000
    MAX_LIST_LENGTH = 10_000
    MAX_ATTRIBUTE_KEY_LENGTH = 200
    MAX_NESTING_DEPTH = 10

    # Dangerous patterns for keys and values
    DANGEROUS_KEY_PATTERNS = [
        "__", "eval", "exec", "compile", "globals", "locals",
        "import", "open", "file", "input", "raw_input", "execfile",
        "reload", "vars", "dir", "delattr", "setattr", "getattr",
        "type", "id", "isinstance", "issubclass", "callable",
        "classmethod", "staticmethod", "property", "super"
    ]

    @classmethod
    def validate_graph_id(cls, graph_id: Any) -> str:
        """Validate graph identifier with strict rules."""
        if not isinstance(graph_id, str):
            msg = f"Graph ID must be string, got {type(graph_id).__name__}"
            raise ValidationError(msg)

        if not graph_id:
            msg = "Graph ID cannot be empty"
            raise ValidationError(msg)

        if len(graph_id) > cls.MAX_IDENTIFIER_LENGTH:
            msg = (
                f"Graph ID too long ({len(graph_id)} chars), "
                f"max {cls.MAX_IDENTIFIER_LENGTH}"
            )
            raise ValidationError(
                msg
            )

        if not cls.GRAPH_ID_PATTERN.match(graph_id):
            msg = (
                "Invalid graph ID format. Must start with alphanumeric, "
                "then alphanumeric, underscore, or hyphen only"
            )
            raise ValidationError(
                msg
            )

        if cls.CONTROL_CHARS.search(graph_id):
            msg = "Graph ID contains control characters"
            raise ValidationError(msg)

        return graph_id

    @classmethod
    def validate_user_id(cls, user_id: Any) -> str:
        """Validate user identifier with strict rules."""
        if not isinstance(user_id, str):
            msg = f"User ID must be string, got {type(user_id).__name__}"
            raise ValidationError(msg)

        if not user_id:
            msg = "User ID cannot be empty"
            raise ValidationError(msg)

        if len(user_id) < 3:
            msg = "User ID must be at least 3 characters"
            raise ValidationError(msg)

        if len(user_id) > 64:
            msg = "User ID too long (max 64 characters)"
            raise ValidationError(msg)

        if not cls.USER_ID_PATTERN.match(user_id):
            msg = (
                "Invalid user ID format. Must be 3-64 chars, "
                "alphanumeric with underscores/hyphens"
            )
            raise ValidationError(
                msg
            )

        if cls.CONTROL_CHARS.search(user_id):
            msg = "User ID contains control characters"
            raise ValidationError(msg)

        return user_id

    @classmethod
    def validate_node_id(cls, node_id: Any) -> Union[str, int, float]:
        """Validate node identifier."""
        if isinstance(node_id, str):
            if not node_id:
                msg = "Node ID cannot be empty string"
                raise ValidationError(msg)

            if len(node_id) > cls.MAX_IDENTIFIER_LENGTH * 10:  # More lenient for nodes
                msg = f"Node ID too long ({len(node_id)} chars), max 1000"
                raise ValidationError(
                    msg
                )

            # Check for dangerous patterns
            if cls.CONTROL_CHARS.search(node_id):
                msg = "Node ID contains control characters"
                raise ValidationError(msg)

            # HTML escape to prevent XSS if displayed
            return html.escape(node_id, quote=False)

        elif isinstance(node_id, (int, float)):
            # Check numeric bounds
            if abs(node_id) > 1e15:
                msg = "Numeric node ID out of safe range"
                raise ValidationError(msg)

            if isinstance(node_id, float) and (
                node_id != node_id or  # NaN check
                node_id == float("inf") or
                node_id == float("-inf")
            ):
                msg = "Node ID cannot be NaN or infinity"
                raise ValidationError(msg)

            return node_id

        else:
            msg = f"Node ID must be string or number, got {type(node_id).__name__}"
            raise ValidationError(
                msg
            )

    @classmethod
    def validate_edge(cls, source: Any, target: Any) -> tuple:
        """Validate edge endpoints."""
        validated_source = cls.validate_node_id(source)
        validated_target = cls.validate_node_id(target)
        return (validated_source, validated_target)

    @classmethod
    def sanitize_attributes(cls, attrs: Any, depth: int = 0) -> Dict[str, Any]:
        """Deep sanitize attributes dictionary."""
        if not isinstance(attrs, dict):
            return {}

        if depth > cls.MAX_NESTING_DEPTH:
            msg = f"Attributes nested too deep (max {cls.MAX_NESTING_DEPTH})"
            raise ValidationError(msg)

        if len(attrs) > cls.MAX_DICT_SIZE:
            msg = f"Too many attributes ({len(attrs)}), max {cls.MAX_DICT_SIZE}"
            raise ValidationError(
                msg
            )

        sanitized = {}

        for key, value in attrs.items():
            # Validate key
            if not isinstance(key, str):
                continue  # Skip non-string keys

            if len(key) > cls.MAX_ATTRIBUTE_KEY_LENGTH:
                continue  # Skip overly long keys

            # Check for dangerous patterns in key
            key_lower = key.lower()
            if any(danger in key_lower for danger in cls.DANGEROUS_KEY_PATTERNS):
                continue  # Skip dangerous keys

            if cls.CONTROL_CHARS.search(key):
                continue  # Skip keys with control chars

            # Sanitize value
            try:
                sanitized_value = cls._sanitize_value(value, depth + 1)
                if sanitized_value is not None:
                    sanitized[key] = sanitized_value
            except ValidationError:
                # Skip values that fail validation
                continue

        return sanitized

    @classmethod
    def _sanitize_value(cls, value: Any, depth: int = 0) -> Any:
        """Recursively sanitize a value."""
        if depth > cls.MAX_NESTING_DEPTH:
            return None

        # Handle None explicitly
        if value is None:
            return None

        # Strings
        if isinstance(value, str):
            if len(value) > cls.MAX_STRING_LENGTH:
                value = value[:cls.MAX_STRING_LENGTH] + "..."

            # Remove control characters
            value = cls.CONTROL_CHARS.sub("", value)

            # HTML escape to prevent XSS
            return html.escape(value, quote=False)

        # Numbers
        elif isinstance(value, bool):
            return value  # Keep booleans as-is

        elif isinstance(value, int):
            # Check bounds
            if abs(value) > 1e15:
                msg = "Integer out of safe range"
                raise ValidationError(msg)
            return value

        elif isinstance(value, float):
            # Check for special values
            if value != value:  # NaN
                return None
            if value == float("inf") or value == float("-inf"):
                return None
            if abs(value) > 1e15:
                msg = "Float out of safe range"
                raise ValidationError(msg)
            return value

        # Lists
        elif isinstance(value, list):
            if len(value) > cls.MAX_LIST_LENGTH:
                value = value[:cls.MAX_LIST_LENGTH]

            sanitized_list = []
            for item in value:
                try:
                    sanitized_item = cls._sanitize_value(item, depth + 1)
                    if sanitized_item is not None:
                        sanitized_list.append(sanitized_item)
                except ValidationError:
                    continue  # Skip invalid items

            return sanitized_list

        # Dictionaries
        elif isinstance(value, dict):
            return cls.sanitize_attributes(value, depth)

        # Dates (convert to ISO string)
        elif isinstance(value, datetime):
            return value.isoformat()

        # Everything else: convert to string with limits
        else:
            str_value = str(value)[:cls.MAX_STRING_LENGTH]
            return html.escape(str_value, quote=False)

    @classmethod
    def validate_graph_type(cls, graph_type: Any) -> str:
        """Validate graph type selection."""
        valid_types = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph"}

        if not isinstance(graph_type, str):
            msg = f"Graph type must be string, got {type(graph_type).__name__}"
            raise ValidationError(msg)

        if graph_type not in valid_types:
            msg = (
                f"Invalid graph type '{graph_type}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            )
            raise ValidationError(
                msg
            )

        return graph_type

    @classmethod
    def validate_file_format(cls, format: Any) -> str:
        """Validate file format for import/export."""
        safe_formats = {
            "graphml", "gml", "pajek", "edgelist",
            "adjlist", "json", "yaml", "gexf"
        }

        if not isinstance(format, str):
            msg = f"Format must be string, got {type(format).__name__}"
            raise ValidationError(msg)

        format_lower = format.lower()

        if format_lower not in safe_formats:
            msg = (
                f"Unsafe format '{format}'. "
                f"Allowed formats: {', '.join(sorted(safe_formats))}"
            )
            raise ValidationError(
                msg
            )

        # Never allow pickle or other executable formats
        dangerous_formats = {"pickle", "pkl", "python", "py", "pyc", "pyo"}
        if format_lower in dangerous_formats:
            msg = f"Format '{format}' is not allowed for security reasons"
            raise SecurityError(msg)

        return format_lower

    @classmethod
    def validate_algorithm_name(cls, algorithm: Any) -> str:
        """Validate algorithm name."""
        if not isinstance(algorithm, str):
            msg = f"Algorithm must be string, got {type(algorithm).__name__}"
            raise ValidationError(msg)

        if not algorithm:
            msg = "Algorithm name cannot be empty"
            raise ValidationError(msg)

        if len(algorithm) > cls.MAX_IDENTIFIER_LENGTH:
            msg = "Algorithm name too long"
            raise ValidationError(msg)

        # Only allow alphanumeric and underscore
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", algorithm):
            msg = (
                "Invalid algorithm name. Must start with letter, "
                "then alphanumeric or underscore only"
            )
            raise ValidationError(
                msg
            )

        return algorithm

    @classmethod
    def sanitize_error_message(cls, error: Exception) -> str:
        """Sanitize error messages to prevent information leakage."""
        error_str = str(error)

        # Remove file paths
        error_str = re.sub(r"(/[^/\s]+)+/([^/\s]+)", "[path]/\\2", error_str)

        # Remove potential secrets (anything that looks like a key/token)
        error_str = re.sub(r"[a-zA-Z0-9]{32,}", "[redacted]", error_str)

        # Limit length
        if len(error_str) > 500:
            error_str = error_str[:500] + "..."

        return error_str
