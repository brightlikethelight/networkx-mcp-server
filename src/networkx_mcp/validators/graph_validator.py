"""Graph validation for NetworkX MCP Server.

This module provides comprehensive validation for graph operations,
ensuring data integrity and security boundaries.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..core.base import Component, ComponentStatus, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class GraphValidationRule:
    """A validation rule for graphs."""

    name: str
    validator: callable
    error_message: str
    is_critical: bool = True


class GraphValidator(Component):
    """Validator for graph operations and data."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__("GraphValidator")
        self.config = config or {}

        # Configuration limits
        self.max_nodes = self.config.get("max_nodes", 1000000)
        self.max_edges = self.config.get("max_edges", 10000000)
        self.max_id_length = self.config.get("max_id_length", 255)
        self.allowed_node_types = self.config.get(
            "allowed_node_types", [str, int, float]
        )
        self.max_attribute_size = self.config.get("max_attribute_size", 1048576)  # 1MB

        # Validation rules
        self._validation_rules: List[GraphValidationRule] = [
            GraphValidationRule(
                name="graph_id_format",
                validator=self._validate_graph_id_format,
                error_message="Graph ID must be alphanumeric with underscores/hyphens",
            ),
            GraphValidationRule(
                name="graph_size_limits",
                validator=self._validate_graph_size_limits,
                error_message=f"Graph exceeds size limits (max {self.max_nodes} nodes, {self.max_edges} edges)",
            ),
            GraphValidationRule(
                name="node_id_format",
                validator=self._validate_node_id_format,
                error_message="Node IDs must be valid and within size limits",
            ),
            GraphValidationRule(
                name="attribute_safety",
                validator=self._validate_attribute_safety,
                error_message="Graph attributes contain unsafe or oversized data",
            ),
        ]

    async def initialize(self) -> None:
        """Initialize the validator."""
        await self._set_status(ComponentStatus.READY)
        logger.info("Graph validator initialized")

    async def validate_graph_creation(
        self, request: Dict[str, Any]
    ) -> ValidationResult:
        """Validate graph creation request."""
        errors = []
        warnings = []

        # Required fields
        required_fields = ["graph_id", "graph_type"]
        for field in required_fields:
            if field not in request:
                errors.append(f"Missing required field: {field}")

        # Graph ID validation
        if "graph_id" in request:
            graph_id = request["graph_id"]
            if not self._validate_graph_id_format(graph_id):
                errors.append("Invalid graph ID format")

        # Graph type validation
        if "graph_type" in request:
            graph_type = request["graph_type"]
            if not self._validate_graph_type(graph_type):
                errors.append(f"Invalid graph type: {graph_type}")

        # Description validation
        if "description" in request:
            description = request["description"]
            if len(description) > 2048:
                warnings.append("Description is very long (>2048 chars)")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"validation_type": "graph_creation"},
        )

    async def validate_graph_data(self, graph: nx.Graph) -> ValidationResult:
        """Validate graph data comprehensively."""
        errors = []
        warnings = []

        # Run all validation rules
        for rule in self._validation_rules:
            try:
                result = rule.validator(graph)
                if not result:
                    if rule.is_critical:
                        errors.append(rule.error_message)
                    else:
                        warnings.append(rule.error_message)
            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                errors.append(f"Validation error: {rule.name}")

        # Additional graph-specific validations
        if graph.number_of_nodes() == 0:
            warnings.append("Graph has no nodes")

        if graph.number_of_edges() == 0 and graph.number_of_nodes() > 1:
            warnings.append("Graph has nodes but no edges")

        # Check for self-loops in undirected graphs
        if not graph.is_directed():
            self_loops = List[Any](nx.selfloop_edges(graph))
            if self_loops:
                warnings.append(f"Undirected graph has {len(self_loops)} self-loops")

        # Check for isolated nodes
        isolated = List[Any](nx.isolates(graph))
        if isolated:
            warnings.append(f"Graph has {len(isolated)} isolated nodes")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "validation_type": "graph_data",
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "isolated_nodes": len(isolated) if "isolated" in locals() else 0,
            },
        )

    async def validate_node_operation(
        self, operation: Dict[str, Any]
    ) -> ValidationResult:
        """Validate node operations (add/remove/update)."""
        errors = []
        warnings = []

        operation_type = operation.get("type")
        if operation_type not in ["add", "remove", "update"]:
            errors.append(f"Invalid operation type: {operation_type}")

        # Validate nodes data
        if "nodes" in operation:
            nodes = operation["nodes"]
            if not isinstance(nodes, List[Any]):
                errors.append("Nodes must be a List[Any]")
            else:
                for i, node in enumerate(nodes):
                    node_errors = self._validate_node_data(node)
                    for error in node_errors:
                        errors.append(f"Node {i}: {error}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"validation_type": "node_operation", "operation": operation_type},
        )

    async def validate_edge_operation(
        self, operation: Dict[str, Any]
    ) -> ValidationResult:
        """Validate edge operations (add/remove/update)."""
        errors = []
        warnings = []

        operation_type = operation.get("type")
        if operation_type not in ["add", "remove", "update"]:
            errors.append(f"Invalid operation type: {operation_type}")

        # Validate edges data
        if "edges" in operation:
            edges = operation["edges"]
            if not isinstance(edges, List[Any]):
                errors.append("Edges must be a List[Any]")
            else:
                for i, edge in enumerate(edges):
                    edge_errors = self._validate_edge_data(edge)
                    for error in edge_errors:
                        errors.append(f"Edge {i}: {error}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"validation_type": "edge_operation", "operation": operation_type},
        )

    def _validate_graph_id_format(self, graph_id: Any) -> bool:
        """Validate graph ID format."""
        if not isinstance(graph_id, str):
            return False

        if len(graph_id) == 0 or len(graph_id) > self.max_id_length:
            return False

        # Allow alphanumeric, underscore, hyphen
        pattern = r"^[a-zA-Z0-9_-]+$"
        return bool(re.match(pattern, graph_id))

    def _validate_graph_type(self, graph_type: str) -> bool:
        """Validate graph type."""
        valid_types = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph"}
        return graph_type in valid_types

    def _validate_graph_size_limits(self, graph: nx.Graph) -> bool:
        """Validate graph size limits."""
        return (
            graph.number_of_nodes() <= self.max_nodes
            and graph.number_of_edges() <= self.max_edges
        )

    def _validate_node_id_format(self, graph: nx.Graph) -> bool:
        """Validate node ID formats."""
        for node in graph.nodes():
            if not self._validate_single_node_id(node):
                return False
        return True

    def _validate_single_node_id(self, node_id: Any) -> bool:
        """Validate a single node ID."""
        # Check type
        if type(node_id) not in self.allowed_node_types:
            return False

        # Check string length
        if isinstance(node_id, str):
            if len(node_id) == 0 or len(node_id) > self.max_id_length:
                return False

        return True

    def _validate_attribute_safety(self, graph: nx.Graph) -> bool:
        """Validate attribute safety and size."""
        # Check graph attributes
        if not self._validate_attributes_dict(graph.graph):
            return False

        # Check node attributes
        for node, attrs in graph.nodes(data=True):
            if not self._validate_attributes_dict(attrs):
                return False

        # Check edge attributes
        for u, v, attrs in graph.edges(data=True):
            if not self._validate_attributes_dict(attrs):
                return False

        return True

    def _validate_attributes_dict(self, attrs: Dict[str, Any]) -> bool:
        """Validate attributes dictionary."""
        if not isinstance(attrs, Dict[str, Any]):
            return True  # Empty attributes are OK

        for key, value in attrs.items():
            # Key validation
            if not isinstance(key, str):
                return False

            if len(key) > 255:
                return False

            # Value validation
            if not self._validate_attribute_value(value):
                return False

        return True

    def _validate_attribute_value(self, value: Any) -> bool:
        """Validate a single attribute value."""
        # Check for unsafe types
        unsafe_types = {type(lambda: None), type(print), type(open)}
        if type(value) in unsafe_types:
            return False

        # Check size for strings and bytes
        if isinstance(value, (str, bytes)):
            if len(value) > self.max_attribute_size:
                return False

        # Recursively check containers
        if isinstance(value, (List[Any], Tuple[Any, ...])):
            if len(value) > 10000:  # Prevent huge lists
                return False
            for item in value:
                if not self._validate_attribute_value(item):
                    return False

        elif isinstance(value, Dict[str, Any]):
            if len(value) > 1000:  # Prevent huge dicts
                return False
            for k, v in value.items():
                if not isinstance(k, (str, int, float)):
                    return False
                if not self._validate_attribute_value(v):
                    return False

        return True

    def _validate_node_data(self, node: Any) -> List[str]:
        """Validate node data."""
        errors = []

        if isinstance(node, Dict[str, Any]):
            # Node with attributes
            if "id" not in node:
                errors.append("Node Dict[str, Any] must have 'id' field")
            else:
                if not self._validate_single_node_id(node["id"]):
                    errors.append("Invalid node ID")

            if "attributes" in node:
                if not self._validate_attributes_dict(node["attributes"]):
                    errors.append("Invalid node attributes")
        else:
            # Simple node ID
            if not self._validate_single_node_id(node):
                errors.append("Invalid node ID")

        return errors

    def _validate_edge_data(self, edge: Any) -> List[str]:
        """Validate edge data."""
        errors = []

        if isinstance(edge, Dict[str, Any]):
            # Edge with attributes
            required_fields = ["source", "target"]
            for field in required_fields:
                if field not in edge:
                    errors.append(f"Edge Dict[str, Any] must have '{field}' field")
                else:
                    if not self._validate_single_node_id(edge[field]):
                        errors.append(f"Invalid {field} ID")

            if "attributes" in edge:
                if not self._validate_attributes_dict(edge["attributes"]):
                    errors.append("Invalid edge attributes")

        elif isinstance(edge, (Tuple[Any, ...], List[Any])):
            # Edge as Tuple[Any, ...]/List[Any]
            if len(edge) < 2:
                errors.append("Edge must have at least source and target")
            else:
                if not self._validate_single_node_id(edge[0]):
                    errors.append("Invalid source node ID")
                if not self._validate_single_node_id(edge[1]):
                    errors.append("Invalid target node ID")

                # Check attributes if present
                if len(edge) > 2:
                    if not self._validate_attributes_dict(edge[2]):
                        errors.append("Invalid edge attributes")
        else:
            errors.append("Edge must be Dict[str, Any], Tuple[Any, ...], or List[Any]")

        return errors

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self.status == ComponentStatus.READY,
            "validation_rules": len(self._validation_rules),
            "config": {
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
                "max_id_length": self.max_id_length,
            },
        }

    def get_validation_rules(self) -> List[str]:
        """Get List[Any] of validation rule names."""
        return [rule.name for rule in self._validation_rules]
