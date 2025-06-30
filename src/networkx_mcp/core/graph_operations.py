"""Core graph operations for NetworkX MCP server."""

from datetime import datetime, timezone
from typing import Any, Optional, Union

import networkx as nx


class GraphManager:
    """Manages NetworkX graph instances and operations."""

    def __init__(self):
        self.graphs: dict[str, nx.Graph] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def create_graph(
        self, graph_id: str, graph_type: str = "Graph", **kwargs
    ) -> dict[str, Any]:
        """Create a new graph instance.

        Args:
            graph_id: Unique identifier for the graph
            graph_type: Type of graph (Graph, DiGraph, MultiGraph, MultiDiGraph)
            **kwargs: Additional graph attributes

        Returns:
            Dict containing graph info and creation status
        """
        if graph_id in self.graphs:
            msg = f"Graph with id '{graph_id}' already exists"
            raise ValueError(msg)

        graph_classes = {
            "Graph": nx.Graph,
            "DiGraph": nx.DiGraph,
            "MultiGraph": nx.MultiGraph,
            "MultiDiGraph": nx.MultiDiGraph,
        }

        if graph_type not in graph_classes:
            msg = f"Invalid graph type: {graph_type}"
            raise ValueError(msg)

        self.graphs[graph_id] = graph_classes[graph_type](**kwargs)
        self.metadata[graph_id] = {
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "graph_type": graph_type,
            "attributes": kwargs,
        }

        return {
            "graph_id": graph_id,
            "graph_type": graph_type,
            "created": True,
            "metadata": self.metadata[graph_id],
        }

    def get_graph(self, graph_id: str) -> nx.Graph:
        """Get a graph by ID."""
        if graph_id not in self.graphs:
            msg = f"Graph '{graph_id}' not found"
            raise KeyError(msg)
        return self.graphs[graph_id]

    def delete_graph(self, graph_id: str) -> dict[str, Any]:
        """Delete a graph by ID."""
        if graph_id not in self.graphs:
            msg = f"Graph '{graph_id}' not found"
            raise KeyError(msg)

        del self.graphs[graph_id]
        del self.metadata[graph_id]

        return {"graph_id": graph_id, "deleted": True}

    def list_graphs(self) -> list[dict[str, Any]]:
        """List all available graphs."""
        return [
            {
                "graph_id": gid,
                "graph_type": self.metadata[gid]["graph_type"],
                "num_nodes": self.graphs[gid].number_of_nodes(),
                "num_edges": self.graphs[gid].number_of_edges(),
                "metadata": self.metadata[gid],
            }
            for gid in self.graphs
        ]

    def add_node(
        self, graph_id: str, node_id: Union[str, int], **attributes
    ) -> dict[str, Any]:
        """Add a node to a graph."""
        graph = self.get_graph(graph_id)
        graph.add_node(node_id, **attributes)

        return {
            "graph_id": graph_id,
            "node_id": node_id,
            "attributes": attributes,
            "added": True,
        }

    def add_nodes_from(
        self, graph_id: str, nodes: list[Union[str, int, tuple]]
    ) -> dict[str, Any]:
        """Add multiple nodes to a graph."""
        graph = self.get_graph(graph_id)
        graph.add_nodes_from(nodes)

        return {
            "graph_id": graph_id,
            "nodes_added": len(nodes),
            "total_nodes": graph.number_of_nodes(),
        }

    def add_edge(
        self,
        graph_id: str,
        source: Union[str, int],
        target: Union[str, int],
        **attributes,
    ) -> dict[str, Any]:
        """Add an edge to a graph."""
        graph = self.get_graph(graph_id)
        graph.add_edge(source, target, **attributes)

        return {
            "graph_id": graph_id,
            "edge": (source, target),
            "attributes": attributes,
            "added": True,
        }

    def add_edges_from(self, graph_id: str, edges: list[tuple]) -> dict[str, Any]:
        """Add multiple edges to a graph."""
        graph = self.get_graph(graph_id)
        graph.add_edges_from(edges)

        return {
            "graph_id": graph_id,
            "edges_added": len(edges),
            "total_edges": graph.number_of_edges(),
        }

    def remove_node(self, graph_id: str, node_id: Union[str, int]) -> dict[str, Any]:
        """Remove a node from a graph."""
        graph = self.get_graph(graph_id)
        if node_id not in graph:
            msg = f"Node '{node_id}' not in graph"
            raise ValueError(msg)

        graph.remove_node(node_id)

        return {"graph_id": graph_id, "node_id": node_id, "removed": True}

    def remove_edge(
        self, graph_id: str, source: Union[str, int], target: Union[str, int]
    ) -> dict[str, Any]:
        """Remove an edge from a graph."""
        graph = self.get_graph(graph_id)
        if not graph.has_edge(source, target):
            msg = f"Edge ({source}, {target}) not in graph"
            raise ValueError(msg)

        graph.remove_edge(source, target)

        return {"graph_id": graph_id, "edge": (source, target), "removed": True}

    def get_graph_info(self, graph_id: str) -> dict[str, Any]:
        """Get detailed information about a graph."""
        graph = self.get_graph(graph_id)

        info = {
            "graph_id": graph_id,
            "graph_type": type(graph).__name__,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "is_directed": graph.is_directed(),
            "is_multigraph": graph.is_multigraph(),
            "density": nx.density(graph),
            "metadata": self.metadata[graph_id],
        }

        if graph.number_of_nodes() > 0:
            info["degree_stats"] = {
                "average": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
                "max": max(dict(graph.degree()).values()) if graph.degree() else 0,
                "min": min(dict(graph.degree()).values()) if graph.degree() else 0,
            }

        return info

    def get_neighbors(
        self, graph_id: str, node_id: Union[str, int]
    ) -> list[Union[str, int]]:
        """Get neighbors of a node."""
        graph = self.get_graph(graph_id)
        if node_id not in graph:
            msg = f"Node '{node_id}' not in graph"
            raise ValueError(msg)

        return list(graph.neighbors(node_id))

    def get_node_attributes(
        self,
        graph_id: str,
        node_id: Optional[Union[str, int]] = None,
        attribute: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get node attributes."""
        graph = self.get_graph(graph_id)

        if node_id is not None:
            if node_id not in graph:
                msg = f"Node '{node_id}' not in graph"
                raise ValueError(msg)
            return graph.nodes[node_id]

        if attribute is not None:
            return nx.get_node_attributes(graph, attribute)

        return dict(graph.nodes(data=True))

    def get_edge_attributes(
        self,
        graph_id: str,
        edge: Optional[tuple[Union[str, int], Union[str, int]]] = None,
        attribute: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get edge attributes."""
        graph = self.get_graph(graph_id)

        if edge is not None:
            if not graph.has_edge(*edge):
                msg = f"Edge {edge} not in graph"
                raise ValueError(msg)
            return graph.edges[edge]

        if attribute is not None:
            return nx.get_edge_attributes(graph, attribute)

        return {edge: data for *edge, data in graph.edges(data=True)}

    def set_node_attributes(
        self,
        graph_id: str,
        values: dict[Union[str, int], dict[str, Any]],
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set node attributes."""
        graph = self.get_graph(graph_id)

        if name is not None:
            nx.set_node_attributes(graph, values, name)
        else:
            nx.set_node_attributes(graph, values)

        return {"graph_id": graph_id, "nodes_updated": len(values), "success": True}

    def set_edge_attributes(
        self,
        graph_id: str,
        values: dict[tuple[Union[str, int], Union[str, int]], dict[str, Any]],
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Set edge attributes."""
        graph = self.get_graph(graph_id)

        if name is not None:
            nx.set_edge_attributes(graph, values, name)
        else:
            nx.set_edge_attributes(graph, values)

        return {"graph_id": graph_id, "edges_updated": len(values), "success": True}

    def subgraph(
        self, graph_id: str, nodes: list[Union[str, int]], create_copy: bool = True
    ) -> Union[nx.Graph, dict[str, Any]]:
        """Create a subgraph from specified nodes."""
        graph = self.get_graph(graph_id)

        if create_copy:
            subgraph = graph.subgraph(nodes).copy()
            return {
                "num_nodes": subgraph.number_of_nodes(),
                "num_edges": subgraph.number_of_edges(),
                "nodes": list(subgraph.nodes()),
                "edges": list(subgraph.edges()),
            }
        else:
            return graph.subgraph(nodes)

    def clear_graph(self, graph_id: str) -> dict[str, Any]:
        """Clear all nodes and edges from a graph."""
        graph = self.get_graph(graph_id)
        graph.clear()

        return {"graph_id": graph_id, "cleared": True, "num_nodes": 0, "num_edges": 0}
