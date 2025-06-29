"""Formatting utilities for graph data."""

import json
from typing import Any, Dict


def format_graph_summary(graph) -> Dict[str, Any]:
    """Format graph summary information."""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "connected": hasattr(graph, 'is_connected') and graph.is_connected() if not graph.is_directed() else None,
        "density": graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2) if graph.number_of_nodes() > 1 else 0
    }

def format_json_output(data: Any, pretty: bool = True) -> str:
    """Format data as JSON string."""
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True, default=str)
    return json.dumps(data, default=str)

def format_error_response(error: Exception, context: str = "") -> Dict[str, str]:
    """Format error as response dict."""
    return {
        "error": str(error),
        "type": type(error).__name__,
        "context": context
    }
