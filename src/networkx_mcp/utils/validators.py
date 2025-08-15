"""Input validation utilities for NetworkX MCP server."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class GraphValidator:
    """Validates graph-related inputs and operations."""

    @staticmethod
    def validate_graph_type(graph_type: str) -> bool:
        """Validate if graph type is supported."""
        valid_types = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph"}
        return graph_type in valid_types

    @staticmethod
    def validate_node_id(node_id: Any) -> bool:
        """Validate node ID format."""
        # None is not valid
        if node_id is None:
            return False

        # Empty strings are not valid
        if isinstance(node_id, str) and not node_id:
            return False

        # Lists and dicts are not valid node IDs
        if isinstance(node_id, (list, dict)):
            return False

        # Node IDs can be non-empty strings, integers, or tuples
        return isinstance(node_id, (str, int, tuple))

    @staticmethod
    def validate_edge(edge: Any) -> bool:
        """Validate edge format."""
        if not isinstance(edge, (tuple, list)):
            return False

        if len(edge) < 2:
            return False

        # Validate source and target nodes
        return GraphValidator.validate_node_id(
            edge[0]
        ) and GraphValidator.validate_node_id(edge[1])

    @staticmethod
    def validate_attributes(attributes: Any) -> bool:
        """Validate node/edge attributes."""
        if not isinstance(attributes, dict):
            return False

        # Check for reserved attribute names
        reserved = {"id", "source", "target", "key"}

        for key in attributes:
            if not isinstance(key, str):
                return False

            # Reserved names are not allowed
            if key in reserved:
                return False

        return True

    @staticmethod
    def validate_weight(graph: nx.Graph, weight: str) -> bool:
        """Validate if weight attribute exists in graph edges."""
        if graph.number_of_edges() == 0:
            return True

        # Check if any edge has the weight attribute
        for _, _, data in graph.edges(data=True):
            if weight in data:
                return True

        return False

    @staticmethod
    def validate_path_exists(
        graph: nx.Graph, source: str | int, target: str | int
    ) -> bool:
        """Validate if path exists between two nodes."""
        if source not in graph or target not in graph:
            return False

        return nx.has_path(graph, source, target)

    @staticmethod
    def validate_graph_connectivity(
        graph: nx.Graph, require_connected: bool = False
    ) -> Dict[str, Any]:
        """Validate graph connectivity properties."""
        result = {
            "valid": True,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }

        if graph.number_of_nodes() == 0:
            result["valid"] = not require_connected
            result["message"] = "Graph is empty"
            return result

        if graph.is_directed():
            result["is_weakly_connected"] = nx.is_weakly_connected(graph)
            result["is_strongly_connected"] = nx.is_strongly_connected(graph)

            if require_connected and not result["is_weakly_connected"]:
                result["valid"] = False
                result["message"] = "Graph is not connected"
        else:
            result["is_connected"] = nx.is_connected(graph)

            if require_connected and not result["is_connected"]:
                result["valid"] = False
                result["message"] = "Graph is not connected"

        return result

    @staticmethod
    def validate_algorithm_input(
        algorithm: str, graph: nx.Graph, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate inputs for specific algorithms."""
        result = {"valid": True, "errors": []}

        # Shortest path algorithms
        if algorithm in ["dijkstra", "bellman-ford", "floyd-warshall"]:
            if "source" in params:
                if params["source"] not in graph:
                    result["valid"] = False
                    result["errors"].append(
                        f"Source node '{params['source']}' not in graph"
                    )

            if "target" in params:
                if params["target"] not in graph:
                    result["valid"] = False
                    result["errors"].append(
                        f"Target node '{params['target']}' not in graph"
                    )

            if "weight" in params:
                if not GraphValidator.validate_weight(graph, params["weight"]):
                    result["errors"].append(
                        f"Weight attribute '{params['weight']}' not found in edges"
                    )

        # Flow algorithms
        elif algorithm in ["maximum_flow", "minimum_cut"]:
            if not graph.is_directed():
                result["valid"] = False
                result["errors"].append("Flow algorithms require directed graphs")

            for node_param in ["source", "sink"]:
                if node_param in params and params[node_param] not in graph:
                    result["valid"] = False
                    result["errors"].append(
                        f"{node_param.capitalize()} node '{params[node_param]}' not in graph"
                    )

        # Spanning tree algorithms
        elif algorithm in ["minimum_spanning_tree", "maximum_spanning_tree"]:
            if graph.is_directed():
                result["valid"] = False
                result["errors"].append(
                    "Spanning tree algorithms require undirected graphs"
                )

            connectivity = GraphValidator.validate_graph_connectivity(
                graph, require_connected=True
            )
            if not connectivity["valid"]:
                result["valid"] = False
                result["errors"].append(connectivity["message"])

        # Coloring algorithms
        elif algorithm == "graph_coloring":
            if graph.is_directed():
                result["errors"].append(
                    "Graph coloring typically applied to undirected graphs"
                )

        return result

    @staticmethod
    def validate_centrality_measure(measure: str) -> bool:
        """Validate centrality measure name."""
        valid_measures = {
            "degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "pagerank",
            "katz",
            "hits",
            "harmonic",
        }
        return measure in valid_measures

    @staticmethod
    def validate_file_format(
        file_format: str | Any, operation: str | List[Any] = "export"
    ) -> bool | Tuple[bool, str | None]:
        """Validate file format for import/export.

        This method supports two signatures for backward compatibility:
        1. validate_file_format(format_str, operation="export") -> bool
        2. validate_file_format(filepath, [formats]) -> Tuple[bool, str | None]
        """
        # Handle the case where it's called with (filepath, [formats])
        if isinstance(operation, list):
            filepath = file_format
            expected_formats = operation
            return GraphValidator.validate_file_path_format(filepath, expected_formats)

        # Original implementation
        if operation == "export":
            valid_formats = {
                "json",
                "graphml",
                "gexf",
                "edgelist",
                "adjacency",
                "pickle",
                "dot",
                "pajek",
                "yaml",
            }
        else:  # import
            valid_formats = {
                "json",
                "graphml",
                "gexf",
                "edgelist",
                "adjacency",
                "pickle",
                "pajek",
                "yaml",
            }

        return file_format.lower() in valid_formats

    @staticmethod
    def validate_layout_algorithm(algorithm: str) -> bool:
        """Validate layout algorithm name."""
        valid_algorithms = {
            "spring",
            "circular",
            "random",
            "shell",
            "spectral",
            "kamada_kawai",
            "planar",
            "fruchterman_reingold",
            "bipartite",
        }
        return algorithm in valid_algorithms

    @staticmethod
    def sanitize_graph_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize graph data for safe processing."""
        sanitized = {}

        # Sanitize nodes
        if "nodes" in data:
            sanitized["nodes"] = []
            for node in data["nodes"]:
                if isinstance(node, dict) and "id" in node:
                    node_data = {"id": str(node["id"])}
                    # Add other attributes
                    for key, value in node.items():
                        if key != "id" and isinstance(key, str):
                            node_data[key] = value
                    sanitized["nodes"].append(node_data)
                else:
                    sanitized["nodes"].append({"id": str(node)})

        # Sanitize edges
        if "edges" in data:
            sanitized["edges"] = []
            for edge in data["edges"]:
                if isinstance(edge, dict):
                    if "source" in edge and "target" in edge:
                        edge_data = {
                            "source": str(edge["source"]),
                            "target": str(edge["target"]),
                        }
                        # Add other attributes
                        for key, value in edge.items():
                            if key not in ["source", "target"] and isinstance(key, str):
                                edge_data[key] = value
                        sanitized["edges"].append(edge_data)
                elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                    sanitized["edges"].append(
                        {"source": str(edge[0]), "target": str(edge[1])}
                    )

        # Copy graph attributes
        if "graph" in data and isinstance(data["graph"], dict):
            sanitized["graph"] = data["graph"].copy()

        return sanitized

    @staticmethod
    def validate_graph_id(graph_id: str) -> Tuple[bool, str | None]:
        """Validate graph ID format and constraints.

        Args:
            graph_id: Graph identifier to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not graph_id:
            return False, "Graph ID cannot be empty"

        if not isinstance(graph_id, str):
            return False, "Graph ID must be a string"

        # Length constraints
        if len(graph_id) < 1 or len(graph_id) > 255:
            return False, "Graph ID must be between 1 and 255 characters"

        # Character constraints - alphanumeric, underscores, hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", graph_id):
            return (
                False,
                "Graph ID must contain only alphanumeric characters, underscores, and hyphens",
            )

        # Reserved names (including Windows reserved names)
        reserved_names = {
            "graph",
            "graphs",
            "List[Any]",
            "all",
            "none",
            "null",
            "undefined",
            "con",
            "prn",
            "aux",
            "nul",
            "com1",
            "com2",
            "com3",
            "com4",
            "com5",
            "com6",
            "com7",
            "com8",
            "com9",
            "lpt1",
            "lpt2",
            "lpt3",
            "lpt4",
            "lpt5",
            "lpt6",
            "lpt7",
            "lpt8",
            "lpt9",
        }
        if graph_id.lower() in reserved_names:
            return False, f"Graph ID '{graph_id}' is reserved"

        return True, None

    @staticmethod
    def validate_file_path_format(
        filepath: str | Path | None, expected_formats: List[str] | None = None
    ) -> Tuple[bool, str | None]:
        """Validate file format based on extension and expected formats.

        Args:
            filepath: Path to file
            expected_formats: List of expected formats (e.g., ['graphml', 'json'])

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle None and empty string cases
        if filepath is None:
            return False, "File path cannot be None"

        if isinstance(filepath, str) and not filepath.strip():
            return False, "File path cannot be empty"

        try:
            path = Path(filepath)

            # Security checks for path traversal
            if ".." in str(filepath):
                return False, "Path traversal attempts are not allowed"

            # Check for URL-like paths
            if isinstance(filepath, str) and (
                "://" in filepath or filepath.startswith("file:")
            ):
                return False, "URL paths are not allowed"

            # Check file exists
            if not path.exists():
                return False, f"File not found: {filepath}"

            if not path.is_file():
                return False, f"Not a file: {filepath}"

            # Check file is readable
            if not path.stat().st_mode & 0o400:
                return False, f"File is not readable: {filepath}"

            # Check file size
            size = path.stat().st_size
            if size == 0:
                return False, f"File is empty: {filepath}"

            # Warn for large files
            if size > 500 * 1024 * 1024:  # 500MB
                logger.warning(
                    f"Large file detected ({size / 1024 / 1024:.1f}MB): {filepath}"
                )

            # Validate format if expected
            if expected_formats:
                ext = path.suffix.lower().lstrip(".")

                # Map extensions to formats
                ext_to_format = {
                    "graphml": "graphml",
                    "xml": "graphml",
                    "gexf": "gexf",
                    "json": "json",
                    "yaml": "yaml",
                    "yml": "yaml",
                    "csv": "csv",
                    "edges": "edgelist",
                    "edgelist": "edgelist",
                    "txt": "edgelist",
                    "adj": "adjacency",
                    "mat": "adjacency",
                    "pickle": "pickle",
                    "pkl": "pickle",
                    "p": "pickle",
                    "net": "pajek",
                    "pajek": "pajek",
                    "dot": "dot",
                    "gv": "dot",
                }

                detected_format = ext_to_format.get(ext)
                if detected_format not in expected_formats:
                    return (
                        False,
                        f"Unexpected file format. Expected {expected_formats}, got '{ext}'",
                    )

            return True, None

        except Exception as e:
            return False, f"Error validating file: {e!s}"

    @staticmethod
    def validate_graph_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate graph data structure for import.

        Args:
            data: Graph data dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not isinstance(data, dict):
            return False, ["Graph data must be a dictionary"]

        # Check for required structure
        if "nodes" in data:
            if not isinstance(data["nodes"], list):
                errors.append("'nodes' must be a List[Any]")
            else:
                # Validate each node
                for i, node in enumerate(data["nodes"]):
                    if isinstance(node, dict):
                        if "id" not in node:
                            errors.append(f"Node at index {i} missing 'id' field")
                        elif not GraphValidator.validate_node_id(node.get("id")):
                            errors.append(
                                f"Invalid node ID at index {i}: {node.get('id')}"
                            )
                    elif not GraphValidator.validate_node_id(node):
                        errors.append(f"Invalid node ID at index {i}: {node}")

        if "edges" in data or "links" in data:
            edges = data.get("edges", data.get("links", []))
            if not isinstance(edges, list):
                errors.append("'edges' must be a List[Any]")
            else:
                # Validate each edge
                for i, edge in enumerate(edges):
                    if isinstance(edge, dict):
                        if "source" not in edge:
                            errors.append(f"Edge at index {i} missing 'source' field")
                        if "target" not in edge:
                            errors.append(f"Edge at index {i} missing 'target' field")
                    elif isinstance(edge, (list, tuple)):
                        if len(edge) < 2:
                            errors.append(
                                f"Edge at index {i} must have at least 2 elements"
                            )
                        elif len(edge) > 3:  # source, target, and optional attributes
                            errors.append(
                                f"Edge at index {i} has too many elements ({len(edge)})"
                            )
                    else:
                        errors.append(f"Invalid edge format at index {i}")

        # Check graph metadata
        if "graph" in data:
            if not isinstance(data["graph"], dict):
                errors.append("'graph' metadata must be a dictionary")

        # Check for alternative formats
        if "adjacency_matrix" in data:
            matrix = data["adjacency_matrix"]
            if not isinstance(matrix, list):
                errors.append("'adjacency_matrix' must be a List[Any]")
            # Check if square matrix
            elif matrix:
                n = len(matrix)
                for i, row in enumerate(matrix):
                    if not isinstance(row, list) or len(row) != n:
                        errors.append(f"Adjacency matrix row {i} has incorrect length")
                        break

        if "edge_list" in data:
            if not isinstance(data["edge_list"], list):
                errors.append("'edge_list' must be a List[Any]")

        return len(errors) == 0, errors

    @staticmethod
    def validate_import_data(
        format: str, data: Any | None = None, path: str | Path | None = None
    ) -> Tuple[bool, str | None]:
        """Validate import data based on format.

        Args:
            format: Import format
            data: Import data (for formats that accept direct data)
            path: File path (for file-based formats)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Formats that require file path
        file_formats = {"graphml", "gexf", "pajek", "dot"}
        # Formats that can accept data or path
        data_formats = {"json", "yaml", "adjacency"}
        # Formats that require path
        path_only_formats = {"csv", "edgelist", "pickle"}

        if format in file_formats or format in path_only_formats:
            if not path:
                return False, f"Format '{format}' requires a file path"

            # Validate file
            valid, error = GraphValidator.validate_file_format(path, [format])
            if not valid:
                return False, error

        elif format in data_formats:
            if not data and not path:
                return False, f"Format '{format}' requires either data or path"

            if path:
                valid, error = GraphValidator.validate_file_format(path, [format])
                if not valid:
                    return False, error

            if data and format in ["json", "yaml", "adjacency"]:
                # Basic structure validation
                if format == "adjacency":
                    if not isinstance(data, dict) or "matrix" not in data:
                        return (
                            False,
                            "Adjacency format requires Dict[str, Any] with 'matrix' key",
                        )

        else:
            return False, f"Unknown import format: {format}"

        return True, None
