"""Output formatting utilities for NetworkX MCP server."""

from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np


class GraphFormatter:
    """Formats graph data for output."""

    @staticmethod
    def format_node_data(
        node_id: Union[str, int],
        attributes: Dict[str, Any],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Format node data for output."""
        result = {
            "id": node_id,
            "attributes": GraphFormatter._serialize_attributes(attributes)
        }

        if include_metadata:
            result["metadata"] = {
                "attribute_count": len(attributes),
                "attribute_keys": list(attributes.keys())
            }

        return result

    @staticmethod
    def format_edge_data(
        source: Union[str, int],
        target: Union[str, int],
        attributes: Dict[str, Any],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Format edge data for output."""
        result = {
            "source": source,
            "target": target,
            "attributes": GraphFormatter._serialize_attributes(attributes)
        }

        if include_metadata:
            result["metadata"] = {
                "attribute_count": len(attributes),
                "attribute_keys": list(attributes.keys()),
                "has_weight": "weight" in attributes
            }

        return result

    @staticmethod
    def format_path(
        path: List[Union[str, int]],
        length: Optional[float] = None,
        include_steps: bool = True
    ) -> Dict[str, Any]:
        """Format path data for output."""
        result = {
            "path": path,
            "num_nodes": len(path),
            "num_edges": len(path) - 1 if len(path) > 1 else 0
        }

        if length is not None:
            result["length"] = length

        if include_steps and len(path) > 1:
            result["steps"] = [
                {"from": path[i], "to": path[i + 1], "step": i + 1}
                for i in range(len(path) - 1)
            ]

        return result

    @staticmethod
    def format_centrality_results(
        centrality_data: Dict[Union[str, int], float],
        measure_name: str,
        top_k: Optional[int] = 10
    ) -> Dict[str, Any]:
        """Format centrality measure results."""
        # Sort by centrality value
        sorted_nodes = sorted(
            centrality_data.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = {
            "measure": measure_name,
            "num_nodes": len(centrality_data),
            "statistics": {
                "mean": np.mean(list(centrality_data.values())),
                "std": np.std(list(centrality_data.values())),
                "min": min(centrality_data.values()) if centrality_data else 0,
                "max": max(centrality_data.values()) if centrality_data else 0
            }
        }

        if top_k and sorted_nodes:
            result["top_nodes"] = [
                {
                    "node": node,
                    "value": value,
                    "rank": i + 1
                }
                for i, (node, value) in enumerate(sorted_nodes[:top_k])
            ]

        result["all_values"] = dict(sorted_nodes)

        return result

    @staticmethod
    def format_community_results(
        communities: List[List[Union[str, int]]],
        modularity: Optional[float] = None,
        include_sizes: bool = True
    ) -> Dict[str, Any]:
        """Format community detection results."""
        result = {
            "num_communities": len(communities),
            "communities": []
        }

        if modularity is not None:
            result["modularity"] = modularity

        for i, community in enumerate(communities):
            comm_data = {
                "id": i,
                "nodes": sorted(community),
                "size": len(community)
            }
            result["communities"].append(comm_data)

        if include_sizes:
            sizes = [len(c) for c in communities]
            result["size_distribution"] = {
                "sizes": sizes,
                "mean": np.mean(sizes),
                "std": np.std(sizes),
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0
            }

        return result

    @staticmethod
    def format_error(
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format error response."""
        result = {
            "error": True,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.now(tz=timezone.utc).isoformat()
        }

        if details:
            result["details"] = details

        return result

    @staticmethod
    def format_success(
        operation: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format success response."""
        result = {
            "success": True,
            "operation": operation,
            "message": message,
            "timestamp": datetime.now(tz=timezone.utc).isoformat()
        }

        if data:
            result["data"] = data

        return result

    @staticmethod
    def format_graph_summary(
        graph_id: str,
        num_nodes: int,
        num_edges: int,
        graph_type: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format graph summary information."""
        result = {
            "graph_id": graph_id,
            "graph_type": graph_type,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        }

        if additional_info:
            result.update(additional_info)

        return result

    @staticmethod
    def format_algorithm_result(
        algorithm: str,
        execution_time: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Format algorithm execution result."""
        result = {
            "algorithm": algorithm,
            "timestamp": datetime.now(tz=timezone.utc).isoformat()
        }

        if execution_time is not None:
            result["execution_time_ms"] = round(execution_time * 1000, 2)

        # Add algorithm-specific results
        result.update(kwargs)

        return result

    @staticmethod
    def _serialize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize attributes for JSON output."""
        serialized = {}

        for key, value in attributes.items():
            if isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, tuple, dict, str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                # Convert to string for unsupported types
                serialized[key] = str(value)

        return serialized

    @staticmethod
    def format_visualization_data(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Format data for graph visualization."""
        result = {
            "nodes": [],
            "edges": []
        }

        # Format nodes with positions
        for node in nodes:
            node_data = {
                "id": node.get("id"),
                "label": str(node.get("label", node.get("id"))),
                "attributes": node.get("attributes", {})
            }

            if layout and node["id"] in layout:
                node_data["x"] = layout[node["id"]][0]
                node_data["y"] = layout[node["id"]][1]

            result["nodes"].append(node_data)

        # Format edges
        for edge in edges:
            edge_data = {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "attributes": edge.get("attributes", {})
            }

            if "weight" in edge_data["attributes"]:
                edge_data["weight"] = edge_data["attributes"]["weight"]

            result["edges"].append(edge_data)

        # Add visualization metadata
        result["metadata"] = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "has_layout": layout is not None
        }

        result["metadata"].update(kwargs)

        return result

    @staticmethod
    def truncate_large_results(
        data: Any,
        max_items: int = 1000,
        max_depth: int = 5
    ) -> Any:
        """Truncate large results to prevent memory issues."""
        def _truncate(obj, depth=0):
            if depth > max_depth:
                return "..."

            if isinstance(obj, dict):
                if len(obj) > max_items:
                    items = list(obj.items())[:max_items]
                    result = dict(items)
                    result["__truncated__"] = f"Showing {max_items} of {len(obj)} items"
                    return result
                else:
                    return {k: _truncate(v, depth + 1) for k, v in obj.items()}

            elif isinstance(obj, list):
                if len(obj) > max_items:
                    result = obj[:max_items]
                    result.append(f"... and {len(obj) - max_items} more items")
                    return result
                else:
                    return [_truncate(item, depth + 1) for item in obj]

            else:
                return obj

        return _truncate(data)
