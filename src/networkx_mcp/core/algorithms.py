"""Graph algorithms implementation for NetworkX MCP server."""

import logging

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)

# Try to import community algorithms - they might not be available in all NetworkX versions
try:
    import networkx.algorithms.community as nx_comm

    from networkx.algorithms.community import modularity
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    nx_comm = None
    modularity = None


class GraphAlgorithms:
    """Implements various graph algorithms using NetworkX."""

    @staticmethod
    def shortest_path(
        graph: nx.Graph,
        source: Union[str, int],
        target: Optional[Union[str, int]] = None,
        weight: Optional[str] = None,
        method: str = "dijkstra"
    ) -> Dict[str, Any]:
        """Find shortest path(s) in a graph."""
        if source not in graph:
            msg = f"Source node '{source}' not in graph"
            raise ValueError(msg)

        if target is not None and target not in graph:
            msg = f"Target node '{target}' not in graph"
            raise ValueError(msg)

        result = {}

        if method == "dijkstra":
            if target:
                path = nx.dijkstra_path(graph, source, target, weight=weight)
                length = nx.dijkstra_path_length(graph, source, target, weight=weight)
                result = {
                    "path": path,
                    "length": length,
                    "source": source,
                    "target": target
                }
            else:
                paths = nx.single_source_dijkstra_path(graph, source, weight=weight)
                lengths = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
                result = {
                    "paths": dict(paths),
                    "lengths": dict(lengths),
                    "source": source
                }
        elif method == "bellman-ford":
            if target:
                path = nx.bellman_ford_path(graph, source, target, weight=weight)
                length = nx.bellman_ford_path_length(graph, source, target, weight=weight)
                result = {
                    "path": path,
                    "length": length,
                    "source": source,
                    "target": target
                }
            else:
                pred, dist = nx.single_source_bellman_ford(graph, source, weight=weight)
                result = {
                    "predecessors": dict(pred),
                    "distances": dict(dist),
                    "source": source
                }
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        return result

    @staticmethod
    def all_pairs_shortest_path(
        graph: nx.Graph,
        weight: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute shortest paths between all pairs of nodes."""
        if weight:
            lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight))
            paths = dict(nx.all_pairs_dijkstra_path(graph, weight=weight))
        else:
            lengths = dict(nx.all_pairs_shortest_path_length(graph))
            paths = dict(nx.all_pairs_shortest_path(graph))

        # Convert to serializable format
        lengths_dict = {str(u): {str(v): length for v, length in d.items()} for u, d in lengths.items()}
        paths_dict = {str(u): {str(v): p for v, p in d.items()} for u, d in paths.items()}

        return {
            "lengths": lengths_dict,
            "paths": paths_dict
        }

    @staticmethod
    def connected_components(graph: nx.Graph) -> Dict[str, Any]:
        """Find connected components in an undirected graph."""
        if graph.is_directed():
            components = list(nx.weakly_connected_components(graph))
            strong_components = list(nx.strongly_connected_components(graph))

            return {
                "weakly_connected_components": [list(comp) for comp in components],
                "strongly_connected_components": [list(comp) for comp in strong_components],
                "num_weakly_connected": len(components),
                "num_strongly_connected": len(strong_components),
                "is_weakly_connected": nx.is_weakly_connected(graph),
                "is_strongly_connected": nx.is_strongly_connected(graph)
            }
        else:
            components = list(nx.connected_components(graph))

            return {
                "connected_components": [list(comp) for comp in components],
                "num_components": len(components),
                "is_connected": nx.is_connected(graph),
                "largest_component_size": max(len(comp) for comp in components) if components else 0
            }

    @staticmethod
    def centrality_measures(
        graph: nx.Graph,
        measures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate various centrality measures."""
        if measures is None:
            measures = ["degree", "betweenness", "closeness", "eigenvector"]

        results = {}

        if "degree" in measures:
            if graph.is_directed():
                results["in_degree_centrality"] = nx.in_degree_centrality(graph)
                results["out_degree_centrality"] = nx.out_degree_centrality(graph)
            else:
                results["degree_centrality"] = nx.degree_centrality(graph)

        if "betweenness" in measures:
            results["betweenness_centrality"] = nx.betweenness_centrality(graph)

        if "closeness" in measures:
            results["closeness_centrality"] = nx.closeness_centrality(graph)

        if "eigenvector" in measures and graph.number_of_edges() > 0:
            try:
                results["eigenvector_centrality"] = nx.eigenvector_centrality(graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                results["eigenvector_centrality"] = {"error": "Failed to converge"}

        if "pagerank" in measures and graph.is_directed():
            results["pagerank"] = nx.pagerank(graph)

        return results

    @staticmethod
    def clustering_coefficients(graph: nx.Graph) -> Dict[str, Any]:
        """Calculate clustering coefficients."""
        if graph.is_directed():
            clustering = nx.clustering(graph.to_undirected())
            avg_clustering = nx.average_clustering(graph.to_undirected())
        else:
            clustering = nx.clustering(graph)
            avg_clustering = nx.average_clustering(graph)

        return {
            "node_clustering": clustering,
            "average_clustering": avg_clustering,
            "transitivity": nx.transitivity(graph)
        }

    @staticmethod
    def minimum_spanning_tree(
        graph: nx.Graph,
        weight: str = "weight",
        algorithm: str = "kruskal"
    ) -> Dict[str, Any]:
        """Find minimum spanning tree."""
        if graph.is_directed():
            msg = "Minimum spanning tree requires undirected graph"
            raise ValueError(msg)

        if algorithm == "kruskal":
            mst = nx.minimum_spanning_tree(graph, weight=weight, algorithm="kruskal")
        elif algorithm == "prim":
            mst = nx.minimum_spanning_tree(graph, weight=weight, algorithm="prim")
        else:
            msg = f"Unknown algorithm: {algorithm}"
            raise ValueError(msg)

        total_weight = sum(data.get(weight, 1) for _, _, data in mst.edges(data=True))

        return {
            "edges": list(mst.edges()),
            "total_weight": total_weight,
            "num_edges": mst.number_of_edges(),
            "algorithm": algorithm
        }

    @staticmethod
    def maximum_flow(
        graph: nx.Graph,
        source: Union[str, int],
        sink: Union[str, int],
        capacity: str = "capacity"
    ) -> Dict[str, Any]:
        """Calculate maximum flow."""
        if not graph.is_directed():
            msg = "Maximum flow requires directed graph"
            raise ValueError(msg)

        flow_value, flow_dict = nx.maximum_flow(graph, source, sink, capacity=capacity)

        return {
            "flow_value": flow_value,
            "flow_dict": dict(flow_dict),
            "source": source,
            "sink": sink
        }

    @staticmethod
    def graph_coloring(
        graph: nx.Graph,
        strategy: str = "largest_first"
    ) -> Dict[str, Any]:
        """Color graph vertices."""
        coloring = nx.greedy_color(graph, strategy=strategy)
        num_colors = max(coloring.values()) + 1 if coloring else 0

        color_classes = defaultdict(list)
        for node, color in coloring.items():
            color_classes[color].append(node)

        return {
            "coloring": coloring,
            "num_colors": num_colors,
            "color_classes": dict(color_classes),
            "chromatic_number_upper_bound": num_colors
        }

    @staticmethod
    def community_detection(
        graph: nx.Graph,
        method: str = "louvain"
    ) -> Dict[str, Any]:
        """Detect communities in a graph."""
        if not HAS_COMMUNITY:
            msg = "Community detection algorithms not available in this NetworkX version"
            raise ImportError(msg)

        if method == "louvain":
            communities = nx_comm.louvain_communities(graph)
        elif method == "label_propagation":
            communities = list(nx_comm.label_propagation_communities(graph))
        elif method == "greedy_modularity":
            communities = list(nx_comm.greedy_modularity_communities(graph))
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        # Convert communities to list format
        communities_list = [list(comm) for comm in communities]

        # Calculate modularity
        mod_score = modularity(graph, communities)

        return {
            "communities": communities_list,
            "num_communities": len(communities_list),
            "modularity": mod_score,
            "method": method,
            "community_sizes": [len(comm) for comm in communities_list]
        }

    @staticmethod
    def cycles_detection(graph: nx.Graph) -> Dict[str, Any]:
        """Detect cycles in a graph."""
        result = {}

        if graph.is_directed():
            has_cycle = not nx.is_directed_acyclic_graph(graph)
            result["has_cycle"] = has_cycle

            if has_cycle:
                try:
                    cycles = list(nx.simple_cycles(graph))
                    result["cycles"] = cycles[:10]  # Limit to first 10 cycles
                    result["num_cycles_found"] = len(cycles)
                except (RecursionError, MemoryError) as e:
                    logger.debug(f"Failed to enumerate cycles: {e}")
                    result["cycles"] = []
                    result["error"] = "Too many cycles to enumerate"

            result["is_dag"] = not has_cycle
        else:
            try:
                cycle_basis = nx.cycle_basis(graph)
                result["cycle_basis"] = cycle_basis
                result["num_independent_cycles"] = len(cycle_basis)
                result["has_cycle"] = len(cycle_basis) > 0
            except Exception as e:
                logger.debug(f"Failed to find cycle basis: {e}")
                result["has_cycle"] = False
                result["cycle_basis"] = []

        return result

    @staticmethod
    def matching(
        graph: nx.Graph,
        max_cardinality: bool = True
    ) -> Dict[str, Any]:
        """Find matching in a graph."""
        if max_cardinality:
            matching = nx.max_weight_matching(graph, maxcardinality=True)
        else:
            matching = nx.maximal_matching(graph)

        matching_list = list(matching)

        return {
            "matching": matching_list,
            "matching_size": len(matching_list),
            "is_perfect": len(matching_list) * 2 == graph.number_of_nodes()
        }

    @staticmethod
    def graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
        """Calculate various graph statistics."""
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed(),
            "is_multigraph": graph.is_multigraph()
        }

        if graph.number_of_nodes() > 0:
            degrees = dict(graph.degree())
            stats["degree_stats"] = {
                "mean": np.mean(list(degrees.values())),
                "std": np.std(list(degrees.values())),
                "min": min(degrees.values()),
                "max": max(degrees.values())
            }

            if graph.is_directed():
                in_degrees = dict(graph.in_degree())
                out_degrees = dict(graph.out_degree())
                stats["in_degree_stats"] = {
                    "mean": np.mean(list(in_degrees.values())),
                    "std": np.std(list(in_degrees.values())),
                    "min": min(in_degrees.values()),
                    "max": max(in_degrees.values())
                }
                stats["out_degree_stats"] = {
                    "mean": np.mean(list(out_degrees.values())),
                    "std": np.std(list(out_degrees.values())),
                    "min": min(out_degrees.values()),
                    "max": max(out_degrees.values())
                }

        # Connectivity
        if graph.is_directed():
            stats["is_weakly_connected"] = nx.is_weakly_connected(graph)
            stats["is_strongly_connected"] = nx.is_strongly_connected(graph)
        else:
            stats["is_connected"] = nx.is_connected(graph)

        # Diameter and radius (only for connected graphs)
        if graph.number_of_nodes() > 0:
            if graph.is_directed():
                if nx.is_strongly_connected(graph):
                    stats["diameter"] = nx.diameter(graph)
                    stats["radius"] = nx.radius(graph)
            elif nx.is_connected(graph):
                stats["diameter"] = nx.diameter(graph)
                stats["radius"] = nx.radius(graph)

        return stats
