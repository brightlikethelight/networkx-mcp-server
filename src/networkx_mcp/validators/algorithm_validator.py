"""Algorithm validation for NetworkX MCP Server.

This module provides validation for algorithm operations,
parameter checking, and computational safety boundaries.
"""

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..core.base import Component, ComponentStatus, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmSpec:
    """Specification for an algorithm."""

    name: str
    required_params: list[str]
    optional_params: dict[str, Any]
    min_nodes: int = 0
    max_nodes: int | None = None
    supports_directed: bool = True
    supports_undirected: bool = True
    supports_weighted: bool = True
    supports_multigraph: bool = False
    computational_complexity: str = "O(n)"  # For warning purposes
    description: str = ""


class AlgorithmValidator(Component):
    """Validator for graph algorithms and their parameters."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("AlgorithmValidator")
        self.config = config or {}

        # Safety limits
        self.max_computation_nodes = self.config.get("max_computation_nodes", 100000)
        self.max_computation_edges = self.config.get("max_computation_edges", 1000000)
        self.timeout_threshold_nodes = self.config.get("timeout_threshold_nodes", 10000)

        # Algorithm specifications
        self._algorithm_specs = {
            # Path algorithms
            "shortest_path": AlgorithmSpec(
                name="shortest_path",
                required_params=["source", "target"],
                optional_params={"weight": None, "method": "dijkstra"},
                computational_complexity="O(n log n + m)",
                description="Find shortest path between two nodes",
            ),
            "all_shortest_paths": AlgorithmSpec(
                name="all_shortest_paths",
                required_params=["source", "target"],
                optional_params={"weight": None},
                max_nodes=10000,  # Expensive for large graphs
                computational_complexity="O(k * (n log n + m))",
                description="Find all shortest paths between two nodes",
            ),
            "shortest_path_length": AlgorithmSpec(
                name="shortest_path_length",
                required_params=[],
                optional_params={"source": None, "target": None, "weight": None},
                computational_complexity="O(n^2)",
                description="Calculate shortest path lengths",
            ),
            # Centrality algorithms
            "degree_centrality": AlgorithmSpec(
                name="degree_centrality",
                required_params=[],
                optional_params={},
                computational_complexity="O(n)",
                description="Calculate degree centrality",
            ),
            "betweenness_centrality": AlgorithmSpec(
                name="betweenness_centrality",
                required_params=[],
                optional_params={"k": None, "normalized": True, "weight": None},
                max_nodes=5000,  # Very expensive
                computational_complexity="O(n^3)",
                description="Calculate betweenness centrality",
            ),
            "closeness_centrality": AlgorithmSpec(
                name="closeness_centrality",
                required_params=[],
                optional_params={"distance": None, "wf_improved": True},
                max_nodes=20000,
                computational_complexity="O(n^2)",
                description="Calculate closeness centrality",
            ),
            "eigenvector_centrality": AlgorithmSpec(
                name="eigenvector_centrality",
                required_params=[],
                optional_params={"max_iter": 100, "tol": 1e-06, "weight": None},
                max_nodes=10000,
                supports_directed=False,  # Requires strong connectivity for directed
                computational_complexity="O(n^2)",
                description="Calculate eigenvector centrality",
            ),
            "pagerank": AlgorithmSpec(
                name="pagerank",
                required_params=[],
                optional_params={
                    "alpha": 0.85,
                    "max_iter": 100,
                    "tol": 1e-06,
                    "weight": None,
                    "dangling": None,
                },
                computational_complexity="O(n * iterations)",
                description="Calculate PageRank",
            ),
            # Community detection
            "connected_components": AlgorithmSpec(
                name="connected_components",
                required_params=[],
                optional_params={},
                computational_complexity="O(n + m)",
                description="Find connected components",
            ),
            "strongly_connected_components": AlgorithmSpec(
                name="strongly_connected_components",
                required_params=[],
                optional_params={},
                supports_undirected=False,
                computational_complexity="O(n + m)",
                description="Find strongly connected components",
            ),
            "weakly_connected_components": AlgorithmSpec(
                name="weakly_connected_components",
                required_params=[],
                optional_params={},
                supports_undirected=False,
                computational_complexity="O(n + m)",
                description="Find weakly connected components",
            ),
            # Clustering
            "clustering": AlgorithmSpec(
                name="clustering",
                required_params=[],
                optional_params={"nodes": None, "weight": None},
                computational_complexity="O(n * d^2)",
                description="Calculate clustering coefficients",
            ),
            "transitivity": AlgorithmSpec(
                name="transitivity",
                required_params=[],
                optional_params={},
                computational_complexity="O(n * d^2)",
                description="Calculate graph transitivity",
            ),
            "triangle_count": AlgorithmSpec(
                name="triangle_count",
                required_params=[],
                optional_params={"nodes": None},
                computational_complexity="O(n * d^2)",
                description="Count triangles in graph",
            ),
            # Structural measures
            "density": AlgorithmSpec(
                name="density",
                required_params=[],
                optional_params={},
                computational_complexity="O(1)",
                description="Calculate graph density",
            ),
            "diameter": AlgorithmSpec(
                name="diameter",
                required_params=[],
                optional_params={},
                max_nodes=5000,  # Expensive for large graphs
                computational_complexity="O(n^2)",
                description="Calculate graph diameter",
            ),
            "radius": AlgorithmSpec(
                name="radius",
                required_params=[],
                optional_params={},
                max_nodes=5000,
                computational_complexity="O(n^2)",
                description="Calculate graph radius",
            ),
            "center": AlgorithmSpec(
                name="center",
                required_params=[],
                optional_params={},
                max_nodes=5000,
                computational_complexity="O(n^2)",
                description="Find graph center nodes",
            ),
            "periphery": AlgorithmSpec(
                name="periphery",
                required_params=[],
                optional_params={},
                max_nodes=5000,
                computational_complexity="O(n^2)",
                description="Find graph periphery nodes",
            ),
            # Flow algorithms
            "maximum_flow": AlgorithmSpec(
                name="maximum_flow",
                required_params=["source", "target"],
                optional_params={"capacity": "capacity"},
                supports_undirected=False,
                computational_complexity="O(n^2 * m)",
                description="Calculate maximum flow",
            ),
            "minimum_cut": AlgorithmSpec(
                name="minimum_cut",
                required_params=["source", "target"],
                optional_params={"capacity": "capacity"},
                supports_undirected=False,
                computational_complexity="O(n^2 * m)",
                description="Calculate minimum cut",
            ),
            # Tree algorithms
            "minimum_spanning_tree": AlgorithmSpec(
                name="minimum_spanning_tree",
                required_params=[],
                optional_params={"weight": "weight", "algorithm": "kruskal"},
                supports_directed=False,
                supports_weighted=True,
                computational_complexity="O(m log n)",
                description="Calculate minimum spanning tree",
            ),
            "maximum_spanning_tree": AlgorithmSpec(
                name="maximum_spanning_tree",
                required_params=[],
                optional_params={"weight": "weight", "algorithm": "kruskal"},
                supports_directed=False,
                supports_weighted=True,
                computational_complexity="O(m log n)",
                description="Calculate maximum spanning tree",
            ),
        }

    async def initialize(self) -> None:
        """Initialize the validator."""
        await self._set_status(ComponentStatus.READY)
        logger.info(
            f"Algorithm validator initialized with {len(self._algorithm_specs)} algorithms"
        )

    async def validate_algorithm_request(self, request: dict[str, Any]) -> ValidationResult:
        """Validate an algorithm execution request."""
        errors = []
        warnings = []

        # Required fields
        required_fields = ["algorithm", "graph_id"]
        for field in required_fields:
            if field not in request:
                errors.append(f"Missing required field: {field}")

        algorithm = request.get("algorithm")
        if not algorithm:
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                metadata={"validation_type": "algorithm_request"},
            )

        # Algorithm existence check
        if algorithm not in self._algorithm_specs:
            errors.append(f"Unknown algorithm: {algorithm}")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                metadata={"validation_type": "algorithm_request"},
            )

        spec = self._algorithm_specs[algorithm]
        parameters = request.get("parameters", {})

        # Parameter validation
        param_errors, param_warnings = self._validate_parameters(spec, parameters)
        errors.extend(param_errors)
        warnings.extend(param_warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "validation_type": "algorithm_request",
                "algorithm": algorithm,
                "complexity": spec.computational_complexity,
            },
        )

    async def validate_algorithm_compatibility(self, algorithm: str, graph: nx.Graph) -> ValidationResult:
        """Validate algorithm compatibility with graph."""
        errors = []
        warnings = []

        if algorithm not in self._algorithm_specs:
            errors.append(f"Unknown algorithm: {algorithm}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        spec = self._algorithm_specs[algorithm]

        # Graph type compatibility
        if graph.is_directed() and not spec.supports_directed:
            errors.append(f"Algorithm {algorithm} does not support directed graphs")

        if not graph.is_directed() and not spec.supports_undirected:
            errors.append(f"Algorithm {algorithm} does not support undirected graphs")

        if graph.is_multigraph() and not spec.supports_multigraph:
            errors.append(f"Algorithm {algorithm} does not support multigraphs")

        # Size limitations
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes < spec.min_nodes:
            errors.append(
                f"Algorithm {algorithm} requires at least {spec.min_nodes} nodes"
            )

        if spec.max_nodes and num_nodes > spec.max_nodes:
            errors.append(
                f"Graph too large for {algorithm} (max {spec.max_nodes} nodes)"
            )

        # Computational safety warnings
        if num_nodes > self.max_computation_nodes:
            errors.append(
                f"Graph exceeds maximum computation limit ({self.max_computation_nodes} nodes)"
            )

        if num_edges > self.max_computation_edges:
            errors.append(
                f"Graph exceeds maximum computation limit ({self.max_computation_edges} edges)"
            )

        # Performance warnings
        if num_nodes > self.timeout_threshold_nodes:
            warnings.append(
                f"Large graph may cause timeout (complexity: {spec.computational_complexity})"
            )

        # Algorithm-specific validations
        algorithm_errors, algorithm_warnings = self._validate_algorithm_specific(
            algorithm, graph
        )
        errors.extend(algorithm_errors)
        warnings.extend(algorithm_warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "validation_type": "algorithm_compatibility",
                "algorithm": algorithm,
                "graph_nodes": num_nodes,
                "graph_edges": num_edges,
            },
        )

    def _validate_parameters(self, spec: AlgorithmSpec, parameters: dict[str, Any]) -> tuple[Any, ...]:
        """Validate algorithm parameters."""
        errors = []
        warnings = []

        # Check required parameters
        for param in spec.required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")

        # Check parameter types and values
        for param, value in parameters.items():
            if param in spec.required_params or param in spec.optional_params:
                # Validate specific parameter types
                param_errors = self._validate_parameter_value(param, value, spec)
                errors.extend(param_errors)
            else:
                warnings.append(f"Unknown parameter: {param}")

        return errors, warnings

    def _validate_parameter_value(self, param: str, value: Any, spec: AlgorithmSpec) -> list[str]:
        """Validate a specific parameter value."""
        errors = []

        # Node ID parameters
        if param in ["source", "target"]:
            if not isinstance(value, (str, int, float)):
                errors.append(f"Parameter {param} must be a valid node ID")

        # Numeric parameters
        elif param in ["alpha", "tol"]:
            if not isinstance(value, (int, float)):
                errors.append(f"Parameter {param} must be numeric")
            elif value <= 0 or value >= 1:
                errors.append(f"Parameter {param} must be between 0 and 1")

        elif param in ["max_iter", "k"]:
            if not isinstance(value, int):
                errors.append(f"Parameter {param} must be an integer")
            elif value <= 0:
                errors.append(f"Parameter {param} must be positive")
            elif value > 10000:
                errors.append(f"Parameter {param} too large (max 10000)")

        # Weight parameter
        elif param == "weight":
            if value is not None and not isinstance(value, str):
                errors.append(
                    "Weight parameter must be a string (attribute name) or None"
                )

        # Method parameter
        elif param == "method":
            if param == "method" and spec.name == "shortest_path":
                valid_methods = ["dijkstra", "bellman_ford"]
                if value not in valid_methods:
                    errors.append(
                        f"Invalid method: {value}. Must be one of {valid_methods}"
                    )

        # Algorithm parameter (for spanning trees)
        elif param == "algorithm":
            if spec.name in ["minimum_spanning_tree", "maximum_spanning_tree"]:
                valid_algorithms = ["kruskal", "prim"]
                if value not in valid_algorithms:
                    errors.append(
                        f"Invalid algorithm: {value}. Must be one of {valid_algorithms}"
                    )

        # Capacity parameter
        elif param == "capacity":
            if not isinstance(value, str):
                errors.append("Capacity parameter must be a string (attribute name)")

        # Nodes parameter (for clustering)
        elif param == "nodes":
            if value is not None:
                if not isinstance(value, list[Any]):
                    errors.append("Nodes parameter must be a list[Any] or None")
                elif len(value) > 10000:
                    errors.append("Too many nodes specified (max 10000)")

        return errors

    def _validate_algorithm_specific(self, algorithm: str, graph: nx.Graph) -> tuple[Any, ...]:
        """Algorithm-specific validations."""
        errors = []
        warnings = []

        # Connectivity requirements
        if algorithm in ["diameter", "radius", "center", "periphery"]:
            if not graph.is_directed() and not nx.is_connected(graph):
                errors.append(f"Algorithm {algorithm} requires connected graph")
            elif graph.is_directed() and not nx.is_strongly_connected(graph):
                errors.append(
                    f"Algorithm {algorithm} requires strongly connected directed graph"
                )

        # Eigenvector centrality specific
        if algorithm == "eigenvector_centrality":
            if graph.is_directed() and not nx.is_strongly_connected(graph):
                warnings.append(
                    "Eigenvector centrality may not converge on non-strongly-connected directed graphs"
                )

        # Flow algorithms
        if algorithm in ["maximum_flow", "minimum_cut"]:
            if not graph.is_directed():
                warnings.append(
                    f"Algorithm {algorithm} works best with directed graphs"
                )

        # Spanning tree algorithms
        if algorithm in ["minimum_spanning_tree", "maximum_spanning_tree"]:
            if not nx.is_connected(graph):
                errors.append(f"Algorithm {algorithm} requires connected graph")

        # Weight-dependent algorithms
        if algorithm in [
            "shortest_path",
            "betweenness_centrality",
            "closeness_centrality",
        ]:
            # Check if graph has negative weights for algorithms that don't support them
            if graph.is_directed():
                for u, v, data in graph.edges(data=True):
                    weight = data.get("weight", 1)
                    if isinstance(weight, (int, float)) and weight < 0:
                        warnings.append(
                            "Graph has negative weights - some algorithms may not work correctly"
                        )
                        break

        return errors, warnings

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self.status == ComponentStatus.READY,
            "supported_algorithms": len(self._algorithm_specs),
            "config": {
                "max_computation_nodes": self.max_computation_nodes,
                "max_computation_edges": self.max_computation_edges,
                "timeout_threshold_nodes": self.timeout_threshold_nodes,
            },
        }

    def get_algorithm_specs(self) -> dict[str, AlgorithmSpec]:
        """Get all algorithm specifications."""
        return self._algorithm_specs.copy()

    def get_supported_algorithms(self) -> list[str]:
        """Get list[Any] of supported algorithms."""
        return list[Any](self._algorithm_specs.keys())

    def get_algorithm_info(self, algorithm: str) -> dict[str, Any] | None:
        """Get detailed information about an algorithm."""
        if algorithm not in self._algorithm_specs:
            return None

        spec = self._algorithm_specs[algorithm]
        return {
            "name": spec.name,
            "description": spec.description,
            "required_params": spec.required_params,
            "optional_params": spec.optional_params,
            "computational_complexity": spec.computational_complexity,
            "supports_directed": spec.supports_directed,
            "supports_undirected": spec.supports_undirected,
            "supports_weighted": spec.supports_weighted,
            "supports_multigraph": spec.supports_multigraph,
            "max_nodes": spec.max_nodes,
        }
