"""Algorithm Handler for NetworkX MCP Server.

This module handles graph algorithm tools including pathfinding,
centrality, connectivity, and other core algorithms.
"""

from typing import Any, Dict, List, Optional, Union
import networkx as nx


try:
    from fastmcp import FastMCP
except ImportError:
    from networkx_mcp.mcp_mock import MockMCP as FastMCP


class AlgorithmHandler:
    """Handler for graph algorithm operations."""

    def __init__(self, mcp: FastMCP, graph_manager):
        """Initialize the handler with MCP server and graph manager."""
        self.mcp = mcp
        self.graph_manager = graph_manager
        self._register_tools()

    def _register_tools(self):
        """Register all algorithm tools."""

        @self.mcp.tool()
        async def shortest_path(
            graph_id: str,
            source: Union[str, int],
            target: Union[str, int],
            weight: Optional[str] = None,
            method: str = "dijkstra",
        ) -> Dict[str, Any]:
            """Find shortest path between two nodes.

            Args:
                graph_id: ID of the graph
                source: Source node
                target: Target node
                weight: Edge attribute to use as weight (optional)
                method: Algorithm to use ('dijkstra', 'bellman-ford', 'astar')

            Returns:
                Dict with path, length, and method used
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if method == "dijkstra" or method == "auto":
                    path = nx.shortest_path(G, source, target, weight=weight)
                    length = nx.shortest_path_length(G, source, target, weight=weight)
                elif method == "bellman-ford":
                    # For negative weights
                    if weight:
                        path = nx.bellman_ford_path(G, source, target, weight=weight)
                        length = nx.bellman_ford_path_length(G, source, target, weight=weight)
                    else:
                        path = nx.shortest_path(G, source, target)
                        length = nx.shortest_path_length(G, source, target)
                elif method == "astar":
                    # Requires heuristic function - use basic for now
                    path = nx.astar_path(G, source, target, weight=weight)
                    length = nx.astar_path_length(G, source, target, weight=weight)
                else:
                    return {"error": f"Unknown method: {method}"}

                return {
                    "path": path,
                    "length": length,
                    "source": source,
                    "target": target,
                    "method": method,
                    "weighted": weight is not None,
                }

            except nx.NetworkXNoPath:
                return {"error": f"No path exists from '{source}' to '{target}'"}
            except nx.NodeNotFound as e:
                return {"error": str(e)}
            except Exception as e:
                return {"error": f"Path finding failed: {str(e)}"}

        @self.mcp.tool()
        async def all_shortest_paths(
            graph_id: str,
            source: Union[str, int],
            target: Union[str, int],
            weight: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Find all shortest paths between two nodes.

            Args:
                graph_id: ID of the graph
                source: Source node
                target: Target node
                weight: Edge attribute to use as weight (optional)

            Returns:
                Dict with all shortest paths and their length
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                paths = list(nx.all_shortest_paths(G, source, target, weight=weight))
                length = nx.shortest_path_length(G, source, target, weight=weight)

                return {
                    "paths": paths,
                    "count": len(paths),
                    "length": length,
                    "source": source,
                    "target": target,
                    "weighted": weight is not None,
                }

            except nx.NetworkXNoPath:
                return {"error": f"No path exists from '{source}' to '{target}'"}
            except nx.NodeNotFound as e:
                return {"error": str(e)}

        @self.mcp.tool()
        async def connected_components(
            graph_id: str, component_type: str = "weak"
        ) -> Dict[str, Any]:
            """Find connected components in a graph.

            Args:
                graph_id: ID of the graph
                component_type: 'weak' or 'strong' (for directed graphs)

            Returns:
                Dict with components and statistics
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if G.is_directed():
                    if component_type == "strong":
                        components = list(nx.strongly_connected_components(G))
                        is_connected = nx.is_strongly_connected(G)
                    else:
                        components = list(nx.weakly_connected_components(G))
                        is_connected = nx.is_weakly_connected(G)
                else:
                    components = list(nx.connected_components(G))
                    is_connected = nx.is_connected(G)

                # Sort components by size
                components = sorted(components, key=len, reverse=True)

                return {
                    "number_of_components": len(components),
                    "is_connected": is_connected,
                    "component_sizes": [len(c) for c in components],
                    "largest_component": list(components[0]) if components else [],
                    "components": [list(c) for c in components[:10]],  # First 10
                    "type": component_type if G.is_directed() else "undirected",
                }

            except Exception as e:
                return {"error": f"Component analysis failed: {str(e)}"}

        @self.mcp.tool()
        async def calculate_centrality(
            graph_id: str,
            centrality_type: str = "degree",
            top_k: int = 10,
            normalized: bool = True,
        ) -> Dict[str, Any]:
            """Calculate node centrality measures.

            Args:
                graph_id: ID of the graph
                centrality_type: Type of centrality ('degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank')
                top_k: Number of top nodes to return
                normalized: Whether to normalize the centrality values

            Returns:
                Dict with centrality scores and top nodes
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                centrality_funcs = {
                    "degree": lambda: nx.degree_centrality(G) if normalized else dict(G.degree()),
                    "betweenness": lambda: nx.betweenness_centrality(G, normalized=normalized),
                    "closeness": lambda: nx.closeness_centrality(G),
                    "eigenvector": lambda: nx.eigenvector_centrality(G, max_iter=1000),
                    "pagerank": lambda: nx.pagerank(G),
                }

                if centrality_type not in centrality_funcs:
                    return {"error": f"Unknown centrality type: {centrality_type}"}

                # Calculate centrality
                centrality = centrality_funcs[centrality_type]()

                # Sort nodes by centrality
                sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

                # Get statistics
                values = list(centrality.values())
                stats = {
                    "mean": sum(values) / len(values) if values else 0,
                    "max": max(values) if values else 0,
                    "min": min(values) if values else 0,
                }

                return {
                    "centrality_type": centrality_type,
                    "top_nodes": sorted_nodes[:top_k],
                    "all_centrality": dict(sorted_nodes) if len(sorted_nodes) <= 100 else None,
                    "statistics": stats,
                    "normalized": normalized,
                }

            except nx.NetworkXError as e:
                return {"error": f"Centrality calculation failed: {str(e)}"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}

        @self.mcp.tool()
        async def clustering_coefficient(
            graph_id: str, nodes: Optional[List[Union[str, int]]] = None
        ) -> Dict[str, Any]:
            """Calculate clustering coefficient.

            Args:
                graph_id: ID of the graph
                nodes: Specific nodes to calculate for (optional)

            Returns:
                Dict with clustering coefficients
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            if G.is_directed():
                return {"error": "Clustering coefficient requires undirected graph"}

            try:
                if nodes:
                    clustering = nx.clustering(G, nodes)
                    avg_clustering = sum(clustering.values()) / len(clustering)
                else:
                    clustering = nx.clustering(G)
                    avg_clustering = nx.average_clustering(G)

                transitivity = nx.transitivity(G)

                return {
                    "average_clustering": avg_clustering,
                    "transitivity": transitivity,
                    "node_clustering": clustering if nodes else None,
                    "number_of_triangles": sum(nx.triangles(G).values()) // 3,
                }

            except Exception as e:
                return {"error": f"Clustering calculation failed: {str(e)}"}

        @self.mcp.tool()
        async def minimum_spanning_tree(
            graph_id: str, weight: str = "weight", algorithm: str = "kruskal"
        ) -> Dict[str, Any]:
            """Find minimum spanning tree.

            Args:
                graph_id: ID of the graph
                weight: Edge attribute to use as weight
                algorithm: 'kruskal', 'prim', or 'boruvka'

            Returns:
                Dict with MST edges and total weight
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            if G.is_directed():
                return {"error": "MST requires undirected graph"}

            try:
                if algorithm == "kruskal":
                    mst = nx.minimum_spanning_tree(G, weight=weight, algorithm="kruskal")
                elif algorithm == "prim":
                    mst = nx.minimum_spanning_tree(G, weight=weight, algorithm="prim")
                elif algorithm == "boruvka":
                    mst = nx.minimum_spanning_tree(G, weight=weight, algorithm="boruvka")
                else:
                    return {"error": f"Unknown algorithm: {algorithm}"}

                edges = list(mst.edges(data=True))
                total_weight = sum(e[2].get(weight, 1) for e in edges)

                return {
                    "edges": [(e[0], e[1], e[2].get(weight, 1)) for e in edges],
                    "total_weight": total_weight,
                    "num_edges": len(edges),
                    "algorithm": algorithm,
                }

            except Exception as e:
                return {"error": f"MST calculation failed: {str(e)}"}

        @self.mcp.tool()
        async def find_cycles(
            graph_id: str, cycle_type: str = "simple"
        ) -> Dict[str, Any]:
            """Find cycles in a graph.

            Args:
                graph_id: ID of the graph
                cycle_type: 'simple' or 'basis'

            Returns:
                Dict with found cycles
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if cycle_type == "simple":
                    if G.is_directed():
                        cycles = list(nx.simple_cycles(G))
                    else:
                        # For undirected, find cycle basis
                        cycles = nx.cycle_basis(G)
                elif cycle_type == "basis":
                    if G.is_directed():
                        return {"error": "Cycle basis is for undirected graphs"}
                    cycles = nx.cycle_basis(G)
                else:
                    return {"error": f"Unknown cycle type: {cycle_type}"}

                # Limit cycles for large graphs
                cycles = cycles[:100]

                return {
                    "num_cycles": len(cycles),
                    "cycles": cycles[:20],  # First 20 cycles
                    "has_cycles": len(cycles) > 0,
                    "cycle_type": cycle_type,
                }

            except Exception as e:
                return {"error": f"Cycle detection failed: {str(e)}"}

        @self.mcp.tool()
        async def topological_sort(graph_id: str) -> Dict[str, Any]:
            """Perform topological sort on a DAG.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with topological ordering
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            if not G.is_directed():
                return {"error": "Topological sort requires directed graph"}

            try:
                if not nx.is_directed_acyclic_graph(G):
                    return {"error": "Graph contains cycles - topological sort not possible"}

                ordering = list(nx.topological_sort(G))

                return {
                    "ordering": ordering,
                    "is_dag": True,
                    "levels": list(nx.topological_generations(G)),
                }

            except Exception as e:
                return {"error": f"Topological sort failed: {str(e)}"}


__all__ = ["AlgorithmHandler"]