"""Formatting utilities for graph data."""

import json
from typing import Any, Dict, Optional


class GraphFormatter:
    """Formatter for graph-related responses."""

    @staticmethod
    def format_success(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Format successful response."""
        return {"success": True, "message": message, "data": data}

    @staticmethod
    def format_error(error_type: str, message: str) -> Dict[str, Any]:
        """Format error response."""
        return {"success": False, "error": error_type, "message": message}

    @staticmethod
    def format_algorithm_result(
        algorithm: str, result: Any, execution_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Format algorithm result."""
        response = {"algorithm": algorithm, "result": result}
        if execution_time is not None:
            response["execution_time_ms"] = execution_time
        return response

    @staticmethod
    def format_community_results(communities: Any) -> Dict[str, Any]:
        """Format community detection results."""
        return {
            "communities": communities,
            "num_communities": (
                len(communities) if hasattr(communities, "__len__") else 0
            ),
        }

    @staticmethod
    def format_visualization_data(
        graph, layout: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format visualization data."""
        nodes = []
        for node in graph.nodes():
            node_data = {"id": str(node), "label": str(node)}
            if layout and node in layout:
                node_data["x"] = layout[node][0]
                node_data["y"] = layout[node][1]
            nodes.append(node_data)

        edges = []
        for u, v, data in graph.edges(data=True):
            edge_data = {"source": str(u), "target": str(v)}
            edge_data.update(data)
            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
            "layout": layout,
            "options": options or {},
        }


def format_graph_summary(graph) -> Dict[str, Any]:
    """Format graph summary information."""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "connected": (
            hasattr(graph, "is_connected") and graph.is_connected()
            if not graph.is_directed()
            else None
        ),
        "density": (
            graph.number_of_edges()
            / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2)
            if graph.number_of_nodes() > 1
            else 0
        ),
    }


def format_json_output(data: Any, pretty: bool = True) -> str:
    """Format data as JSON string."""
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True, default=str)
    return json.dumps(data, default=str)


def format_error_response(error: Exception, context: str = "") -> Dict[str, str]:
    """Format error as response dict."""
    return {"error": str(error), "type": type(error).__name__, "context": context}
