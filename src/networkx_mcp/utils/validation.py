"""Input validation utilities."""

import re
from typing import Any, Dict, List, Union

class InputValidator:
    """Validate inputs for graph operations."""
    
    GRAPH_ID_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$')
    NODE_ID_PATTERN = re.compile(r'^[^<>&"\']{1,1000}$')
    
    @classmethod
    def validate_graph_id(cls, graph_id: str) -> str:
        """Validate graph ID format."""
        if not isinstance(graph_id, str) or not cls.GRAPH_ID_PATTERN.match(graph_id):
            raise ValueError(f"Invalid graph ID: {graph_id}")
        return graph_id
    
    @classmethod
    def validate_node_id(cls, node_id: Union[str, int]) -> Union[str, int]:
        """Validate node ID format."""
        if isinstance(node_id, str):
            if not cls.NODE_ID_PATTERN.match(node_id):
                raise ValueError(f"Invalid node ID: {node_id}")
        elif not isinstance(node_id, (int, float)):
            raise ValueError(f"Node ID must be string or number: {type(node_id)}")
        return node_id
    
    @classmethod
    def sanitize_attributes(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize node/edge attributes."""
        sanitized = {}
        for key, value in attrs.items():
            # Remove dangerous keys
            if key.startswith('_') or key in ['eval', 'exec', '__']:
                continue
            # Sanitize values
            if isinstance(value, str) and len(value) > 10000:
                value = value[:10000]  # Truncate long strings
            sanitized[key] = value
        return sanitized
