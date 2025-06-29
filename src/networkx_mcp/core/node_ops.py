"""Focused node operations module."""

from typing import Any, Dict, List, Union

import networkx as nx


class NodeOperations:
    """Handle all node-related operations efficiently."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def add_node_with_validation(self, node_id: Union[str, int], **attrs) -> bool:
        """Add node with validation."""
        if node_id in self.graph:
            return False
        self.graph.add_node(node_id, **attrs)
        return True

    def bulk_add_nodes(self, nodes: List[Union[str, int, tuple]]) -> int:
        """Efficiently add multiple nodes."""
        initial_count = self.graph.number_of_nodes()
        self.graph.add_nodes_from(nodes)
        return self.graph.number_of_nodes() - initial_count

    def get_node_summary(self, node_id: Union[str, int]) -> Dict[str, Any]:
        """Get comprehensive node information."""
        if node_id not in self.graph:
            msg = f"Node {node_id} not found"
            raise ValueError(msg)

        return {
            "id": node_id,
            "attributes": dict(self.graph.nodes[node_id]),
            "degree": self.graph.degree(node_id),
            "neighbors": list(self.graph.neighbors(node_id))
        }
