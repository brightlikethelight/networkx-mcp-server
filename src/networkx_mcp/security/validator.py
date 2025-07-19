"""Security validation utilities."""

import re
from typing import Any


class SecurityError(Exception):
    """Base security exception."""


class SecurityValidator:
    """Security validation for user inputs."""

    # Patterns for validation
    ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")

    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate user ID format."""
        if not user_id:
            raise SecurityError("User ID cannot be empty")

        if not isinstance(user_id, str):
            raise SecurityError("User ID must be a string")

        if not SecurityValidator.ID_PATTERN.match(user_id):
            raise SecurityError(f"Invalid user ID format: {user_id}")

        return user_id

    @staticmethod
    def validate_graph_id(graph_id: str) -> str:
        """Validate graph ID format."""
        if not graph_id:
            raise SecurityError("Graph ID cannot be empty")

        if not isinstance(graph_id, str):
            raise SecurityError("Graph ID must be a string")

        if not SecurityValidator.ID_PATTERN.match(graph_id):
            raise SecurityError(f"Invalid graph ID format: {graph_id}")

        return graph_id

    @staticmethod
    def sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
        """Sanitize user-provided attributes."""
        if not isinstance(attributes, dict):
            return {}

        # Remove any potentially dangerous keys
        dangerous_keys = ["__", "eval", "exec", "compile", "globals", "locals"]

        sanitized: dict[str, Any] = {}
        for key, value in attributes.items():
            # Check key
            if not isinstance(key, str):
                continue

            # Skip dangerous keys
            if any(danger in key.lower() for danger in dangerous_keys):
                continue

            # Limit key length
            if len(key) > 100:
                continue

            # Sanitize value (basic type checking)
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Only allow simple lists
                if all(
                    isinstance(item, (str, int, float, bool, type(None)))
                    for item in value[:100]
                ):
                    # Create a list from the first 100 items
                    limited_list = list(value[:100])
                    sanitized[key] = limited_list
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts (with depth limit)
                if len(str(value)) < 10000:  # Limit total size
                    # Recursively sanitize nested dict
                    nested_sanitized = SecurityValidator.sanitize_attributes(value)
                    sanitized[key] = nested_sanitized

        return sanitized
