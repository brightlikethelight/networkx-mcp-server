"""Focused edge operations module."""

from typing import Any, Union

import networkx as nx


class EdgeOperations:
    """Handle all edge-related operations efficiently."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def add_edge_with_validation(
        self, source: Union[str, int], target: Union[str, int], **attrs
    ) -> bool:
        """Add edge with validation."""
        if self.graph.has_edge(source, target):
            return False
        self.graph.add_edge(source, target, **attrs)
        return True

    def bulk_add_edges(self, edges: list[tuple]) -> int:
        """Efficiently add multiple edges."""
        initial_count = self.graph.number_of_edges()
        self.graph.add_edges_from(edges)
        return self.graph.number_of_edges() - initial_count

    def get_edge_summary(
        self, source: Union[str, int], target: Union[str, int]
    ) -> dict[str, Any]:
        """Get comprehensive edge information."""
        if not self.graph.has_edge(source, target):
            msg = f"Edge ({source}, {target}) not found"
            raise ValueError(msg)

        return {
            "source": source,
            "target": target,
            "attributes": dict(self.graph.edges[source, target]),
            "weight": self.graph.edges[source, target].get("weight", 1),
        }
