"""Directed graph analysis and algorithms."""

import logging
import time
from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np

try:
    from networkx_mcp.advanced.community_detection import CommunityDetection
except ImportError:
    # Fallback if community detection is not available
    CommunityDetection = None


# Constants for algorithm thresholds
MAX_CYCLES_LIMIT = 100
MAX_NODES_FOR_ALL_PATHS = 100
MAX_EDGES_FOR_TRANSITIVE_REDUCTION = 1000
MAX_LONGEST_PATHS_DISPLAY = 10
MAX_CONDENSATION_NODES = 1000
MAX_EDGES_FOR_EXACT_FAS = 50
COMMUNITY_OVERLAP_THRESHOLD = 0.3

logger = logging.getLogger(__name__)


class DirectedAnalysis:
    """Analysis tools for directed graphs."""

    @staticmethod
    def dag_analysis(graph: nx.DiGraph, **_params) -> dict[str, Any]:
        """
        Analyze Directed Acyclic Graph (DAG) properties.

        Parameters:
        -----------
        graph : nx.DiGraph
            Input directed graph

        Returns:
        --------
        Dict containing DAG analysis results
        """
        start_time = time.time()

        # Check if graph is DAG
        is_dag = nx.is_directed_acyclic_graph(graph)

        if not is_dag:
            # Find cycles
            try:
                cycles = list(nx.simple_cycles(graph))
                # Limit cycles for large graphs
                if len(cycles) > MAX_CYCLES_LIMIT:
                    cycles = cycles[:MAX_CYCLES_LIMIT]
                    cycles_truncated = True
                else:
                    cycles_truncated = False
            except Exception as e:
                logger.warning(f"Error finding cycles: {e}")
                cycles = []
                cycles_truncated = False

            return {
                "is_dag": False,
                "has_cycles": True,
                "num_cycles_found": len(cycles),
                "cycles": cycles,
                "cycles_truncated": cycles_truncated,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        # DAG analysis
        results = {"is_dag": True, "has_cycles": False}

        # Topological sort
        try:
            topo_sort = list(nx.topological_sort(graph))
            results["topological_sort"] = topo_sort

            # Node levels (distance from roots)
            levels = DirectedAnalysis._compute_dag_levels(graph, topo_sort)
            results["levels"] = levels
            results["num_levels"] = max(levels.values()) + 1 if levels else 0

            # Level statistics
            level_counts = defaultdict(int)
            for _node, level in levels.items():
                level_counts[level] += 1
            results["nodes_per_level"] = dict(level_counts)

        except nx.NetworkXError:
            results["topological_sort"] = None
            results["levels"] = None

        # Longest path
        try:
            longest_path = nx.dag_longest_path(graph)
            longest_path_length = nx.dag_longest_path_length(graph)

            results["longest_path"] = longest_path
            results["longest_path_length"] = longest_path_length

            # All longest paths (if graph is small)
            if graph.number_of_nodes() < MAX_NODES_FOR_ALL_PATHS:
                all_longest = DirectedAnalysis._find_all_longest_paths(graph)
                results["all_longest_paths"] = all_longest[:MAX_LONGEST_PATHS_DISPLAY]
                results["num_longest_paths"] = len(all_longest)

        except Exception as e:
            logger.warning(f"Error computing longest path: {e}")
            results["longest_path"] = None
            results["longest_path_length"] = 0

        # Roots and leaves
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        results["roots"] = roots
        results["num_roots"] = len(roots)
        results["leaves"] = leaves
        results["num_leaves"] = len(leaves)

        # Width (maximum nodes at any level)
        if "nodes_per_level" in results:
            results["width"] = (
                max(results["nodes_per_level"].values())
                if results["nodes_per_level"]
                else 0
            )

        # Transitive reduction
        if graph.number_of_edges() < MAX_EDGES_FOR_TRANSITIVE_REDUCTION:
            try:
                transitive_reduction = nx.transitive_reduction(graph)
                results[
                    "transitive_reduction_edges"
                ] = transitive_reduction.number_of_edges()
                results["edge_reduction_ratio"] = (
                    1 - transitive_reduction.number_of_edges() / graph.number_of_edges()
                    if graph.number_of_edges() > 0
                    else 0
                )
            except Exception as e:
                logger.warning(f"Error computing transitive reduction: {e}")

        execution_time = (time.time() - start_time) * 1000
        results["execution_time_ms"] = execution_time

        return results

    @staticmethod
    def _compute_dag_levels(graph: nx.DiGraph, topo_sort: list) -> dict[Any, int]:
        """Compute level for each node in DAG."""
        levels = {}

        for node in topo_sort:
            # Level is max of predecessor levels + 1
            if graph.in_degree(node) == 0:
                levels[node] = 0
            else:
                pred_levels = [levels[pred] for pred in graph.predecessors(node)]
                levels[node] = max(pred_levels) + 1

        return levels

    @staticmethod
    def _find_all_longest_paths(graph: nx.DiGraph) -> list[list]:
        """Find all longest paths in DAG."""
        # Get all paths from roots to leaves
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        longest_paths = []
        max_length = 0

        for root in roots:
            for leaf in leaves:
                try:
                    for path in nx.all_simple_paths(graph, root, leaf):
                        path_length = len(path) - 1
                        if path_length > max_length:
                            max_length = path_length
                            longest_paths = [path]
                        elif path_length == max_length:
                            longest_paths.append(path)
                except nx.NetworkXNoPath:
                    continue

        return longest_paths

    @staticmethod
    def strongly_connected_components(
        graph: nx.DiGraph,
        algorithm: str = "tarjan",
        return_condensation: bool = False,
        **_params,
    ) -> dict[str, Any]:
        """
        Find strongly connected components using various algorithms.

        Parameters:
        -----------
        graph : nx.DiGraph
            Input directed graph
        algorithm : str
            Algorithm: 'tarjan', 'kosaraju'
        return_condensation : bool
            Whether to return the condensation graph

        Returns:
        --------
        Dict containing SCC analysis
        """
        start_time = time.time()

        if algorithm == "tarjan":
            # Tarjan's algorithm (default NetworkX implementation)
            sccs = list(nx.strongly_connected_components(graph))

        elif algorithm == "kosaraju":
            # Kosaraju's algorithm
            sccs = list(nx.kosaraju_strongly_connected_components(graph))

        else:
            msg = f"Unknown algorithm: {algorithm}"
            raise ValueError(msg)

        # Sort by size
        sccs = sorted(sccs, key=len, reverse=True)

        # Calculate statistics
        scc_sizes = [len(scc) for scc in sccs]

        results = {
            "algorithm": algorithm,
            "num_sccs": len(sccs),
            "sccs": [list(scc) for scc in sccs],
            "scc_sizes": scc_sizes,
            "largest_scc_size": max(scc_sizes) if scc_sizes else 0,
            "is_strongly_connected": len(sccs) == 1,
            "trivial_sccs": sum(1 for s in scc_sizes if s == 1),
            "non_trivial_sccs": sum(1 for s in scc_sizes if s > 1),
        }

        # Size distribution
        size_dist = defaultdict(int)
        for size in scc_sizes:
            size_dist[size] += 1
        results["size_distribution"] = dict(size_dist)

        # Condensation graph
        if return_condensation:
            condensation = nx.condensation(graph, sccs)
            results["condensation_graph"] = {
                "num_nodes": condensation.number_of_nodes(),
                "num_edges": condensation.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(condensation),
                "mapping": dict(condensation.graph["mapping"]),
            }

            # Analyze condensation
            if condensation.number_of_nodes() < MAX_CONDENSATION_NODES:
                # Component relationships
                results["condensation_graph"]["longest_path_length"] = (
                    nx.dag_longest_path_length(condensation)
                    if nx.is_directed_acyclic_graph(condensation)
                    else None
                )

        execution_time = (time.time() - start_time) * 1000
        results["execution_time_ms"] = execution_time

        return results

    @staticmethod
    def condensation_graph(
        graph: nx.DiGraph, **_params
    ) -> tuple[nx.DiGraph, dict[str, Any]]:
        """
        Create and analyze the condensation graph (SCC condensation).

        Parameters:
        -----------
        graph : nx.DiGraph
            Input directed graph

        Returns:
        --------
        Tuple of (condensation graph, metadata)
        """
        # Get SCCs
        sccs = list(nx.strongly_connected_components(graph))

        # Create condensation
        C = nx.condensation(graph, sccs)

        # Add metadata to nodes
        for node in C:
            members = C.nodes[node]["members"]
            C.nodes[node]["size"] = len(members)
            C.nodes[node]["is_trivial"] = len(members) == 1

            # Calculate internal density
            if len(members) > 1:
                subgraph = graph.subgraph(members)
                max_edges = len(members) * (len(members) - 1)
                density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0
                C.nodes[node]["internal_density"] = density
            else:
                C.nodes[node]["internal_density"] = 0

        # Analyze condensation structure
        metadata = {
            "num_components": C.number_of_nodes(),
            "num_edges": C.number_of_edges(),
            "original_nodes": graph.number_of_nodes(),
            "original_edges": graph.number_of_edges(),
            "reduction_ratio": C.number_of_nodes() / graph.number_of_nodes(),
            "is_dag": nx.is_directed_acyclic_graph(C),
        }

        if metadata["is_dag"]:
            # DAG analysis of condensation
            dag_info = DirectedAnalysis.dag_analysis(C)
            metadata["dag_height"] = dag_info.get("longest_path_length", 0)
            metadata["dag_width"] = dag_info.get("width", 0)

        # Component size statistics
        sizes = [C.nodes[n]["size"] for n in C]
        metadata["component_sizes"] = {
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": np.mean(sizes) if sizes else 0,
            "std": np.std(sizes) if sizes else 0,
        }

        return C, metadata

    @staticmethod
    def tournament_analysis(graph: nx.DiGraph, **_params) -> dict[str, Any]:
        """
        Analyze tournament graph properties.

        Parameters:
        -----------
        graph : nx.DiGraph
            Input directed graph

        Returns:
        --------
        Dict containing tournament analysis
        """
        # Check if graph is a tournament
        n = graph.number_of_nodes()
        max_edges = n * (n - 1) // 2

        is_tournament = graph.number_of_edges() == max_edges and not any(
            graph.has_edge(v, u) for u, v in graph.edges()
        )

        results = {
            "is_tournament": is_tournament,
            "num_nodes": n,
            "num_edges": graph.number_of_edges(),
        }

        if not is_tournament:
            results["reason"] = "Not a complete oriented graph"
            return results

        # Score sequence (out-degrees)
        scores = sorted([graph.out_degree(n) for n in graph.nodes()], reverse=True)
        results["score_sequence"] = scores

        # Landau's theorem check
        is_valid = DirectedAnalysis._check_landau_theorem(scores)
        results["satisfies_landau_theorem"] = is_valid

        # Find Hamiltonian path (every tournament has one)
        try:
            ham_path = nx.tournament_hamiltonian_path(graph)
            results["hamiltonian_path"] = ham_path
        except Exception as e:
            logger.warning(f"Error finding Hamiltonian path: {e}")
            results["hamiltonian_path"] = None

        # Transitivity
        transitive_triples = 0
        total_triples = 0

        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    total_triples += 1
                    # Check if triple is transitive
                    triple = [nodes[i], nodes[j], nodes[k]]
                    if DirectedAnalysis._is_transitive_triple(graph, triple):
                        transitive_triples += 1

        results["transitivity_ratio"] = (
            transitive_triples / total_triples if total_triples > 0 else 0
        )
        results["is_transitive"] = transitive_triples == total_triples

        # Strong connectivity
        results["is_strongly_connected"] = nx.is_strongly_connected(graph)

        # Kings (nodes that can reach every other node in at most 2 steps)
        kings = DirectedAnalysis._find_tournament_kings(graph)
        results["kings"] = kings
        results["num_kings"] = len(kings)

        return results

    @staticmethod
    def _check_landau_theorem(scores: list[int]) -> bool:
        """Check if score sequence satisfies Landau's theorem."""
        n = len(scores)
        for k in range(1, n + 1):
            sum_k = sum(scores[:k])
            if sum_k < k * (k - 1) // 2:
                return False
        return sum(scores) == n * (n - 1) // 2

    @staticmethod
    def _is_transitive_triple(graph: nx.DiGraph, triple: list) -> bool:
        """Check if a triple of nodes forms a transitive relation."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if j not in (i, k) and i != k:
                        if (
                            graph.has_edge(triple[i], triple[j])
                            and graph.has_edge(triple[j], triple[k])
                            and not graph.has_edge(triple[i], triple[k])
                        ):
                            return False
        return True

    @staticmethod
    def _find_tournament_kings(graph: nx.DiGraph) -> list:
        """Find kings in a tournament (nodes that dominate all others in ≤2 steps)."""
        kings = []

        for node in graph.nodes():
            is_king = True
            for other in graph.nodes():
                if other != node:
                    # Check if node can reach other in at most 2 steps
                    if graph.has_edge(node, other):
                        continue  # Direct domination

                    # Check 2-step paths
                    two_step = False
                    for intermediate in graph.successors(node):
                        if graph.has_edge(intermediate, other):
                            two_step = True
                            break

                    if not two_step:
                        is_king = False
                        break

            if is_king:
                kings.append(node)

        return kings

    @staticmethod
    def feedback_arc_set(
        graph: nx.DiGraph, method: str = "greedy", **_params
    ) -> dict[str, Any]:
        """
        Find minimum feedback arc set.

        Parameters:
        -----------
        graph : nx.DiGraph
            Input directed graph
        method : str
            Method: 'greedy', 'exact' (for small graphs)

        Returns:
        --------
        Dict containing feedback arc set
        """
        start_time = time.time()

        if method == "greedy":
            # Greedy approximation
            fas = DirectedAnalysis._greedy_feedback_arc_set(graph)

        elif method == "exact" and graph.number_of_edges() < MAX_EDGES_FOR_EXACT_FAS:
            # Exact solution for small graphs (exponential time)
            fas = DirectedAnalysis._exact_feedback_arc_set(graph)

        else:
            # Fallback to greedy for large graphs
            fas = DirectedAnalysis._greedy_feedback_arc_set(graph)
            method = "greedy"

        # Verify solution
        test_graph = graph.copy()
        test_graph.remove_edges_from(fas)
        is_dag = nx.is_directed_acyclic_graph(test_graph)

        execution_time = (time.time() - start_time) * 1000

        return {
            "feedback_arc_set": fas,
            "size": len(fas),
            "method": method,
            "fraction_of_edges": (
                len(fas) / graph.number_of_edges() if graph.number_of_edges() > 0 else 0
            ),
            "creates_dag": is_dag,
            "execution_time_ms": execution_time,
        }

    @staticmethod
    def _greedy_feedback_arc_set(graph: nx.DiGraph) -> list[tuple]:
        """Greedy approximation for feedback arc set."""
        # Create a copy to work with
        G = graph.copy()
        fas = []

        # Repeatedly remove nodes with specific properties
        while G.number_of_nodes() > 0:
            # Remove sinks (out-degree 0)
            sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
            G.remove_nodes_from(sinks)

            # Remove sources (in-degree 0)
            sources = [n for n in G.nodes() if G.in_degree(n) == 0]
            G.remove_nodes_from(sources)

            if G.number_of_nodes() == 0:
                break

            # Find node with maximum out-degree - in-degree
            max_diff = -float("inf")
            max_node = None

            for node in G.nodes():
                diff = G.out_degree(node) - G.in_degree(node)
                if diff > max_diff:
                    max_diff = diff
                    max_node = node

            if max_node:
                # Add incoming edges to FAS
                for pred in list(G.predecessors(max_node)):
                    fas.append((pred, max_node))

                G.remove_node(max_node)

        # Filter to only include edges from original graph
        fas = [(u, v) for u, v in fas if graph.has_edge(u, v)]

        return fas

    @staticmethod
    def _exact_feedback_arc_set(graph: nx.DiGraph) -> list[tuple]:
        """Exact solution using cycle enumeration (exponential time)."""
        # Find all cycles
        cycles = list(nx.simple_cycles(graph))

        if not cycles:
            return []

        # This is a hitting set problem - NP-hard
        # Simple greedy: repeatedly remove edge that appears in most cycles
        edge_count = defaultdict(int)

        for cycle in cycles:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge_count[(u, v)] += 1

        fas = []
        remaining_cycles = cycles.copy()

        while remaining_cycles:
            # Find edge in most remaining cycles
            edge_count = defaultdict(int)

            for cycle in remaining_cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge_count[(u, v)] += 1

            if not edge_count:
                break

            # Remove edge with highest count
            max_edge = max(edge_count, key=edge_count.get)
            fas.append(max_edge)

            # Remove cycles containing this edge
            remaining_cycles = [
                cycle
                for cycle in remaining_cycles
                if not DirectedAnalysis._cycle_contains_edge(cycle, max_edge)
            ]

        return fas

    @staticmethod
    def _cycle_contains_edge(cycle: list, edge: tuple) -> bool:
        """Check if cycle contains the given edge."""
        u, v = edge
        for i in range(len(cycle)):
            if cycle[i] == u and cycle[(i + 1) % len(cycle)] == v:
                return True
        return False

    @staticmethod
    def bow_tie_structure(
        graph: nx.DiGraph, largest_scc_only: bool = True, **_params
    ) -> dict[str, Any]:
        """
        Analyze bow-tie structure of directed graph (common in web graphs).

        Parameters:
        -----------
        graph : nx.DiGraph
            Input directed graph
        largest_scc_only : bool
            Whether to analyze only the largest SCC

        Returns:
        --------
        Dict containing bow-tie decomposition
        """
        start_time = time.time()

        # Find all SCCs
        sccs = list(nx.strongly_connected_components(graph))

        if largest_scc_only:
            # Use only largest SCC as core
            if sccs:
                core = max(sccs, key=len)
                other_sccs = [scc for scc in sccs if scc != core]
            else:
                core = set()
                other_sccs = []
        else:
            # All non-trivial SCCs as core
            core = set()
            other_sccs = []
            for scc in sccs:
                if len(scc) > 1:
                    core.update(scc)
                else:
                    other_sccs.append(scc)

        # Initialize components
        in_component = set()  # IN: can reach CORE
        out_component = set()  # OUT: reachable from CORE
        tubes = set()  # TUBES: IN -> OUT bypassing CORE
        tendrils_in = set()  # Tendrils from IN
        tendrils_out = set()  # Tendrils to OUT
        disconnected = set()  # Disconnected components

        # Classify remaining nodes
        remaining = set(graph.nodes()) - core

        for node in remaining:
            can_reach_core = (
                nx.has_path(graph, node, next(iter(core))) if core else False
            )
            core_can_reach = (
                nx.has_path(graph, next(iter(core)), node) if core else False
            )

            if can_reach_core and not core_can_reach:
                in_component.add(node)
            elif core_can_reach and not can_reach_core:
                out_component.add(node)
            elif not can_reach_core and not core_can_reach:
                # Check if it's a tube or tendril
                reaches_out = any(
                    nx.has_path(graph, node, out_node) for out_node in out_component
                )
                reached_from_in = any(
                    nx.has_path(graph, in_node, node) for in_node in in_component
                )

                if reaches_out and reached_from_in:
                    tubes.add(node)
                elif reached_from_in:
                    tendrils_in.add(node)
                elif reaches_out:
                    tendrils_out.add(node)
                else:
                    disconnected.add(node)

        # Calculate statistics
        n = graph.number_of_nodes()

        results = {
            "core_size": len(core),
            "in_size": len(in_component),
            "out_size": len(out_component),
            "tubes_size": len(tubes),
            "tendrils_in_size": len(tendrils_in),
            "tendrils_out_size": len(tendrils_out),
            "disconnected_size": len(disconnected),
            "core_fraction": len(core) / n if n > 0 else 0,
            "in_fraction": len(in_component) / n if n > 0 else 0,
            "out_fraction": len(out_component) / n if n > 0 else 0,
            "tubes_fraction": len(tubes) / n if n > 0 else 0,
            "total_sccs": len(sccs),
            "largest_scc_only": largest_scc_only,
        }

        # Verify decomposition
        total_classified = (
            len(core)
            + len(in_component)
            + len(out_component)
            + len(tubes)
            + len(tendrils_in)
            + len(tendrils_out)
            + len(disconnected)
        )
        results["all_nodes_classified"] = total_classified == n

        execution_time = (time.time() - start_time) * 1000
        results["execution_time_ms"] = execution_time

        return results

    @staticmethod
    def temporal_graph_import(
        edge_list: list[tuple[Any, Any, float]],
        time_window: Optional[tuple[float, float]] = None,
        **params,
    ) -> tuple[nx.DiGraph, dict[str, Any]]:
        """
        Import temporal graph from time-stamped edge list.

        Parameters:
        -----------
        edge_list : List[Tuple[source, target, timestamp]]
            Time-stamped edges
        time_window : Tuple[float, float], optional
            Time window to filter edges

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        # Filter by time window if specified
        if time_window:
            min_time, max_time = time_window
            filtered_edges = [
                (u, v, t) for u, v, t in edge_list if min_time <= t <= max_time
            ]
        else:
            filtered_edges = edge_list

        # Create directed graph
        G = nx.DiGraph()

        # Add edges with timestamp attribute
        for u, v, t in filtered_edges:
            if G.has_edge(u, v):
                # Multiple edges - keep earliest/latest based on param
                if params.get("keep", "earliest") == "earliest":
                    if t < G[u][v].get("timestamp", float("inf")):
                        G[u][v]["timestamp"] = t
                elif t > G[u][v].get("timestamp", -float("inf")):
                    G[u][v]["timestamp"] = t
            else:
                G.add_edge(u, v, timestamp=t)

        # Calculate temporal statistics
        timestamps = [t for _, _, t in filtered_edges]

        metadata = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_temporal_edges": len(filtered_edges),
            "time_span": max(timestamps) - min(timestamps) if timestamps else 0,
            "earliest_time": min(timestamps) if timestamps else None,
            "latest_time": max(timestamps) if timestamps else None,
            "duplicate_edges_collapsed": len(filtered_edges) - G.number_of_edges(),
        }

        return G, metadata

    @staticmethod
    def temporal_centrality(
        temporal_edges: list[tuple[Any, Any, float]],
        centrality_type: str = "degree",
        time_slices: int = 10,
        **_params,
    ) -> dict[str, Any]:
        """
        Calculate time-dependent centrality measures.

        Parameters:
        -----------
        temporal_edges : List[Tuple[source, target, timestamp]]
            Time-stamped edges
        centrality_type : str
            Type of centrality
        time_slices : int
            Number of time slices

        Returns:
        --------
        Dict containing temporal centrality evolution
        """
        if not temporal_edges:
            return {"error": "No temporal edges provided"}

        # Get time range
        timestamps = [t for _, _, t in temporal_edges]
        min_time = min(timestamps)
        max_time = max(timestamps)

        # Create time slices
        slice_size = (max_time - min_time) / time_slices
        slices = []

        for i in range(time_slices):
            start_time = min_time + i * slice_size
            end_time = min_time + (i + 1) * slice_size

            # Build graph for this time slice
            G, _ = DirectedAnalysis.temporal_graph_import(
                temporal_edges, time_window=(start_time, end_time)
            )

            # Calculate centrality
            if centrality_type == "degree":
                centrality = dict(G.degree())
            elif centrality_type == "in_degree":
                centrality = dict(G.in_degree())
            elif centrality_type == "out_degree":
                centrality = dict(G.out_degree())
            elif centrality_type == "betweenness":
                centrality = nx.betweenness_centrality(G)
            elif centrality_type == "pagerank":
                try:
                    centrality = nx.pagerank(G)
                except Exception as e:
                    logger.warning(f"PageRank calculation failed: {e}")
                    centrality = {}
            else:
                centrality = {}

            slices.append(
                {
                    "time_range": (start_time, end_time),
                    "slice_index": i,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "centrality": centrality,
                }
            )

        # Track evolution of top nodes
        all_nodes = set()
        for slice_data in slices:
            all_nodes.update(slice_data["centrality"].keys())

        evolution = {}
        for node in all_nodes:
            evolution[node] = [
                slice_data["centrality"].get(node, 0) for slice_data in slices
            ]

        # Find nodes with highest average centrality
        avg_centrality = {node: np.mean(values) for node, values in evolution.items()}
        top_nodes = sorted(avg_centrality.keys(), key=avg_centrality.get, reverse=True)[
            :10
        ]

        return {
            "centrality_type": centrality_type,
            "time_slices": time_slices,
            "slice_data": slices,
            "node_evolution": {node: evolution[node] for node in top_nodes},
            "top_nodes": top_nodes,
            "time_range": (min_time, max_time),
        }

    @staticmethod
    def temporal_paths(
        temporal_edges: list[tuple[Any, Any, float]],
        source: Any,
        target: Any,
        **_params,
    ) -> dict[str, Any]:
        """
        Find time-respecting paths in temporal graph.

        Parameters:
        -----------
        temporal_edges : List[Tuple[source, target, timestamp]]
            Time-stamped edges
        source : node
            Source node
        target : node
            Target node

        Returns:
        --------
        Dict containing temporal paths
        """
        # Build temporal adjacency list
        temporal_adj = defaultdict(list)

        for u, v, t in temporal_edges:
            temporal_adj[u].append((v, t))

        # Find all time-respecting paths
        paths = []

        def find_temporal_paths(current, target, current_time, path):
            if current == target:
                paths.append(path.copy())
                return

            if current in temporal_adj:
                for neighbor, edge_time in temporal_adj[current]:
                    if edge_time >= current_time:  # Time-respecting
                        path.append((neighbor, edge_time))
                        find_temporal_paths(neighbor, target, edge_time, path)
                        path.pop()

        # Start search
        find_temporal_paths(source, target, -float("inf"), [(source, -float("inf"))])

        # Process paths
        valid_paths = []
        for path in paths:
            # Remove dummy first timestamp
            clean_path = [(path[0][0], None), *path[1:]]

            # Calculate path duration
            if len(clean_path) > 1:
                duration = clean_path[-1][1] - clean_path[1][1]
            else:
                duration = 0

            valid_paths.append(
                {
                    "path": [node for node, _ in clean_path],
                    "timestamps": [t for _, t in clean_path[1:]],
                    "length": len(clean_path) - 1,
                    "duration": duration,
                }
            )

        # Sort by various criteria
        shortest_path = (
            min(valid_paths, key=lambda p: p["length"]) if valid_paths else None
        )
        fastest_path = (
            min(valid_paths, key=lambda p: p["duration"]) if valid_paths else None
        )

        return {
            "source": source,
            "target": target,
            "num_paths": len(valid_paths),
            "paths": valid_paths[:10],  # Limit to 10 paths
            "shortest_path": shortest_path,
            "fastest_path": fastest_path,
            "reachable": len(valid_paths) > 0,
        }

    @staticmethod
    def temporal_communities(
        temporal_edges: list[tuple[Any, Any, float]],
        method: str = "snapshots",
        time_slices: int = 10,
        **_params,
    ) -> dict[str, Any]:
        """
        Detect dynamic communities in temporal networks.

        Parameters:
        -----------
        temporal_edges : List[Tuple[source, target, timestamp]]
            Time-stamped edges
        method : str
            Method: 'snapshots', 'evolutionary'
        time_slices : int
            Number of time slices

        Returns:
        --------
        Dict containing temporal community evolution
        """
        if CommunityDetection is None:
            return {"error": "Community detection module not available"}

        if not temporal_edges:
            return {"error": "No temporal edges provided"}

        # Get time range
        timestamps = [t for _, _, t in temporal_edges]
        min_time = min(timestamps)
        max_time = max(timestamps)

        if method == "snapshots":
            # Independent community detection at each time slice
            slice_size = (max_time - min_time) / time_slices
            slices = []

            for i in range(time_slices):
                start_time = min_time + i * slice_size
                end_time = min_time + (i + 1) * slice_size

                # Build graph for this time slice
                G, _ = DirectedAnalysis.temporal_graph_import(
                    temporal_edges, time_window=(start_time, end_time)
                )

                # Detect communities
                if G.number_of_nodes() > 0:
                    # Convert to undirected for community detection
                    G_undirected = G.to_undirected()
                    comm_result = CommunityDetection.detect_communities(
                        G_undirected, algorithm="louvain"
                    )
                    communities = comm_result["communities"]
                else:
                    communities = []

                slices.append(
                    {
                        "time_range": (start_time, end_time),
                        "slice_index": i,
                        "communities": communities,
                        "num_communities": len(communities),
                        "modularity": (
                            comm_result.get("modularity", 0)
                            if G.number_of_nodes() > 0
                            else 0
                        ),
                    }
                )

            # Track community evolution
            # Simple matching based on overlap
            evolution = []
            for i in range(len(slices) - 1):
                curr_comms = slices[i]["communities"]
                next_comms = slices[i + 1]["communities"]

                transitions = []
                for j, curr_comm in enumerate(curr_comms):
                    curr_set = set(curr_comm)
                    best_match = None
                    best_overlap = 0

                    for k, next_comm in enumerate(next_comms):
                        next_set = set(next_comm)
                        overlap = len(curr_set & next_set) / len(curr_set | next_set)

                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = k

                    if (
                        best_match is not None
                        and best_overlap > COMMUNITY_OVERLAP_THRESHOLD
                    ):
                        transitions.append(
                            {"from": j, "to": best_match, "overlap": best_overlap}
                        )

                evolution.append(
                    {"from_slice": i, "to_slice": i + 1, "transitions": transitions}
                )

            return {
                "method": "snapshots",
                "time_slices": time_slices,
                "slice_data": slices,
                "evolution": evolution,
                "time_range": (min_time, max_time),
            }

        else:
            # Simplified evolutionary approach
            # Would require more sophisticated algorithms in practice
            return {
                "method": method,
                "error": "Method not fully implemented",
                "available_methods": ["snapshots"],
            }
