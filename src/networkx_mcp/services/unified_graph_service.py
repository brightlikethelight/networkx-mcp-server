"""Simple Unified Graph Service bridging GraphManager and GraphAlgorithms.

This service provides a consistent API that accepts graph IDs for all operations,
eliminating the need for users to manually bridge between GraphManager and GraphAlgorithms.
"""

import logging
from typing import Any

from ..core.algorithms import GraphAlgorithms
from ..core.graph_operations import GraphManager

logger = logging.getLogger(__name__)


class UnifiedGraphService:
    """Simple unified service for graph operations and algorithms.

    Provides a consistent API that accepts graph IDs for all operations,
    eliminating the need for users to manually bridge between GraphManager
    and GraphAlgorithms.
    """

    def __init__(self):
        self.graph_manager = GraphManager()
        self.algorithms = GraphAlgorithms()

    # === Graph Management Operations ===

    def create_graph(
        self, graph_id: str, graph_type: str = "Graph", **kwargs
    ) -> dict[str, Any]:
        """Create a new graph."""
        try:
            result = self.graph_manager.create_graph(graph_id, graph_type, **kwargs)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def delete_graph(self, graph_id: str) -> dict[str, Any]:
        """Delete a graph."""
        try:
            result = self.graph_manager.delete_graph(graph_id)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def list_graphs(self) -> dict[str, Any]:
        """List all graphs."""
        try:
            graphs = self.graph_manager.list_graphs()
            return {"status": "success", "graphs": graphs, "count": len(graphs)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_graph_info(self, graph_id: str) -> dict[str, Any]:
        """Get information about a graph."""
        try:
            result = self.graph_manager.get_graph_info(graph_id)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    # === Node Operations ===

    def add_node(
        self, graph_id: str, node_id: str | int, **attributes
    ) -> dict[str, Any]:
        """Add a single node to a graph."""
        try:
            result = self.graph_manager.add_node(graph_id, node_id, **attributes)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def add_nodes(self, graph_id: str, nodes: list[str | int]) -> dict[str, Any]:
        """Add multiple nodes to a graph."""
        try:
            result = self.graph_manager.add_nodes_from(graph_id, nodes)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def remove_node(self, graph_id: str, node_id: str | int) -> dict[str, Any]:
        """Remove a node from a graph."""
        try:
            result = self.graph_manager.remove_node(graph_id, node_id)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def get_node_attributes(self, graph_id: str, node_id: str | int) -> dict[str, Any]:
        """Get attributes of a node."""
        try:
            attrs = self.graph_manager.get_node_attributes(graph_id, node_id)
            return {"status": "success", "node_id": node_id, "attributes": attrs}
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    # === Edge Operations ===

    def add_edge(
        self, graph_id: str, source: str | int, target: str | int, **attributes
    ) -> dict[str, Any]:
        """Add a single edge to a graph."""
        try:
            result = self.graph_manager.add_edge(graph_id, source, target, **attributes)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def add_edges(
        self, graph_id: str, edges: list[tuple[str | int, str | int]]
    ) -> dict[str, Any]:
        """Add multiple edges to a graph."""
        try:
            result = self.graph_manager.add_edges_from(graph_id, edges)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def remove_edge(
        self, graph_id: str, source: str | int, target: str | int
    ) -> dict[str, Any]:
        """Remove an edge from a graph."""
        try:
            result = self.graph_manager.remove_edge(graph_id, source, target)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def get_edge_attributes(
        self, graph_id: str, source: str | int, target: str | int
    ) -> dict[str, Any]:
        """Get attributes of an edge."""
        try:
            attrs = self.graph_manager.get_edge_attributes(graph_id, (source, target))
            return {"status": "success", "edge": (source, target), "attributes": attrs}
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def get_neighbors(self, graph_id: str, node_id: str | int) -> dict[str, Any]:
        """Get neighbors of a node."""
        try:
            neighbors = self.graph_manager.get_neighbors(graph_id, node_id)
            return {"status": "success", "node_id": node_id, "neighbors": neighbors}
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    # === Algorithm Operations (Unified API) ===

    def shortest_path(
        self,
        graph_id: str,
        source: str | int,
        target: str | int | None = None,
        weight: str | None = None,
        method: str = "dijkstra",
    ) -> dict[str, Any]:
        """Find shortest path in a graph."""
        try:
            graph = self.graph_manager.get_graph(graph_id)
            result = self.algorithms.shortest_path(
                graph, source, target, weight, method
            )
            result["status"] = "success"
            result["graph_id"] = graph_id
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def connected_components(self, graph_id: str) -> dict[str, Any]:
        """Find connected components in a graph."""
        try:
            graph = self.graph_manager.get_graph(graph_id)
            result = self.algorithms.connected_components(graph)
            result["status"] = "success"
            result["graph_id"] = graph_id
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def centrality_measures(
        self, graph_id: str, measures: list[str] | None = None
    ) -> dict[str, Any]:
        """Calculate centrality measures for a graph."""
        try:
            graph = self.graph_manager.get_graph(graph_id)
            result = self.algorithms.centrality_measures(graph, measures)
            result["status"] = "success"
            result["graph_id"] = graph_id
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def clustering_coefficients(self, graph_id: str) -> dict[str, Any]:
        """Calculate clustering coefficients for a graph."""
        try:
            graph = self.graph_manager.get_graph(graph_id)
            result = self.algorithms.clustering_coefficients(graph)
            result["status"] = "success"
            result["graph_id"] = graph_id
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def minimum_spanning_tree(
        self, graph_id: str, weight: str = "weight", algorithm: str = "kruskal"
    ) -> dict[str, Any]:
        """Find minimum spanning tree of a graph."""
        try:
            graph = self.graph_manager.get_graph(graph_id)
            result = self.algorithms.minimum_spanning_tree(graph, weight, algorithm)
            result["status"] = "success"
            result["graph_id"] = graph_id
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def community_detection(
        self, graph_id: str, method: str = "louvain"
    ) -> dict[str, Any]:
        """Detect communities in a graph."""
        try:
            graph = self.graph_manager.get_graph(graph_id)
            result = self.algorithms.community_detection(graph, method)
            result["status"] = "success"
            result["graph_id"] = graph_id
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    # === Utility Operations ===

    def subgraph(
        self,
        graph_id: str,
        nodes: list[str | int],
        new_graph_id: str | None = None,
        create_copy: bool = True,
    ) -> dict[str, Any]:
        """Create a subgraph from specified nodes.

        If new_graph_id is provided, creates a new managed graph with that ID.
        Otherwise, returns subgraph information without creating a new graph.
        """
        try:
            if new_graph_id:
                # Create a new managed graph with the subgraph
                source_graph = self.graph_manager.get_graph(graph_id)
                subgraph_nx = (
                    source_graph.subgraph(nodes).copy()
                    if create_copy
                    else source_graph.subgraph(nodes)
                )

                # Determine graph type
                graph_type = type(source_graph).__name__

                # Create new graph and copy data
                self.graph_manager.create_graph(new_graph_id, graph_type)
                target_graph = self.graph_manager.get_graph(new_graph_id)

                # Copy nodes and edges
                target_graph.add_nodes_from(subgraph_nx.nodes(data=True))
                target_graph.add_edges_from(subgraph_nx.edges(data=True))

                return {
                    "status": "success",
                    "graph_id": graph_id,
                    "subgraph_id": new_graph_id,
                    "num_nodes": target_graph.number_of_nodes(),
                    "num_edges": target_graph.number_of_edges(),
                    "nodes": list(target_graph.nodes()),
                    "edges": list(target_graph.edges()),
                }
            else:
                # Just return subgraph information
                result = self.graph_manager.subgraph(graph_id, nodes, create_copy)
                result["status"] = "success"
                result["graph_id"] = graph_id
                return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}

    def clear_graph(self, graph_id: str) -> dict[str, Any]:
        """Clear all nodes and edges from a graph."""
        try:
            result = self.graph_manager.clear_graph(graph_id)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": str(e), "graph_id": graph_id}
