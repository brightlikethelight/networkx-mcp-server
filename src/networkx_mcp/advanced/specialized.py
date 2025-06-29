"""Specialized graph algorithms and advanced analysis."""

import logging
import random  # Using for non-cryptographic algorithm simulation only
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SpecializedAlgorithms:
    """Advanced and specialized graph algorithms."""

    @staticmethod
    def spanning_trees(
        graph: Union[nx.Graph, nx.DiGraph],
        algorithm: str = "kruskal",
        weight: Optional[str] = "weight",
        k: int = 1,
        **params
    ) -> Dict[str, Any]:
        """
        Find minimum/maximum spanning trees using various algorithms.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        algorithm : str
            Algorithm: 'kruskal', 'prim', 'boruvka', 'steiner'
        weight : str
            Edge attribute for weights
        k : int
            Number of spanning trees to find (for k-MST)

        Returns:
        --------
        Dict containing spanning tree(s) and metadata
        """
        start_time = time.time()

        # Convert directed to undirected if needed
        if graph.is_directed():
            G = graph.to_undirected()
        else:
            G = graph.copy()

        # Check connectivity
        if not nx.is_connected(G):
            # Find MST for each component
            components = list(nx.connected_components(G))
            forest = []
            total_weight = 0

            for component in components:
                subgraph = G.subgraph(component)
                if subgraph.number_of_edges() > 0:
                    mst = nx.minimum_spanning_tree(subgraph, weight=weight, algorithm=algorithm)
                    forest.extend(mst.edges(data=True))
                    total_weight += sum(d.get(weight, 1) for _, _, d in mst.edges(data=True))

            return {
                "is_tree": False,
                "is_forest": True,
                "num_components": len(components),
                "edges": forest,
                "total_weight": total_weight,
                "algorithm": algorithm,
                "execution_time_ms": (time.time() - start_time) * 1000
            }

        results = {"algorithm": algorithm}

        if algorithm in ["kruskal", "prim", "boruvka"]:
            # Standard MST algorithms
            mst = nx.minimum_spanning_tree(G, weight=weight, algorithm=algorithm)

            edges = list(mst.edges(data=True))
            total_weight = sum(d.get(weight, 1) for _, _, d in edges)

            results.update({
                "edges": edges,
                "num_edges": len(edges),
                "total_weight": total_weight,
                "is_tree": True
            })

            # Find k-MST if requested
            if k > 1:
                k_trees = SpecializedAlgorithms._find_k_spanning_trees(G, k, weight)
                results["k_spanning_trees"] = k_trees
                results["num_trees_found"] = len(k_trees)

        elif algorithm == "steiner":
            # Steiner tree (approximation)
            terminals = params.get("terminals", [])
            if not terminals:
                # Use random subset as terminals
                num_terminals = max(2, G.number_of_nodes() // 3)
                terminals = random.sample(list(G.nodes()), num_terminals)

            steiner = nx.approximation.steiner_tree(G, terminals, weight=weight)

            edges = list(steiner.edges(data=True))
            total_weight = sum(d.get(weight, 1) for _, _, d in edges)

            results.update({
                "edges": edges,
                "num_edges": len(edges),
                "total_weight": total_weight,
                "terminals": terminals,
                "num_terminals": len(terminals),
                "num_steiner_nodes": steiner.number_of_nodes() - len(terminals),
                "approximation_ratio": 2.0  # Known approximation ratio
            })

        else:
            msg = f"Unknown algorithm: {algorithm}"
            raise ValueError(msg)

        # Add statistics
        if "edges" in results:
            edge_weights = [d.get(weight, 1) for _, _, d in results["edges"]]
            if edge_weights:
                results["weight_statistics"] = {
                    "min": min(edge_weights),
                    "max": max(edge_weights),
                    "mean": np.mean(edge_weights),
                    "std": np.std(edge_weights)
                }

        results["execution_time_ms"] = (time.time() - start_time) * 1000

        return results

    @staticmethod
    def _find_k_spanning_trees(
        graph: nx.Graph,
        k: int,
        weight: str
    ) -> List[Dict[str, Any]]:
        """Find k different spanning trees."""
        trees = []
        used_edge_sets = set()

        # First MST
        mst = nx.minimum_spanning_tree(graph, weight=weight)
        mst_edges = frozenset(mst.edges())
        used_edge_sets.add(mst_edges)

        trees.append({
            "edges": list(mst.edges(data=True)),
            "weight": sum(d.get(weight, 1) for _, _, d in mst.edges(data=True))
        })

        # Try to find more by excluding edges
        attempts = 0
        max_attempts = k * 10

        while len(trees) < k and attempts < max_attempts:
            attempts += 1

            # Randomly exclude some edges
            exclude_edges = random.sample(
                list(graph.edges()),
                min(graph.number_of_edges() // 4, 5)
            )

            temp_graph = graph.copy()
            temp_graph.remove_edges_from(exclude_edges)

            if nx.is_connected(temp_graph):
                mst = nx.minimum_spanning_tree(temp_graph, weight=weight)
                mst_edges = frozenset(mst.edges())

                if mst_edges not in used_edge_sets:
                    used_edge_sets.add(mst_edges)
                    trees.append({
                        "edges": list(mst.edges(data=True)),
                        "weight": sum(d.get(weight, 1) for _, _, d in mst.edges(data=True))
                    })

        return trees

    @staticmethod
    def graph_coloring(
        graph: nx.Graph,
        strategy: str = "greedy",
        **params
    ) -> Dict[str, Any]:
        """
        Color graph nodes using various strategies.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        strategy : str
            Strategy: 'greedy', 'welsh_powell', 'dsatur', 'brooks'

        Returns:
        --------
        Dict containing coloring and statistics
        """
        start_time = time.time()

        if strategy == "greedy":
            # Greedy coloring with various orderings
            ordering = params.get("ordering", "largest_first")

            if ordering == "largest_first":
                node_ordering = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
            elif ordering == "smallest_last":
                node_ordering = SpecializedAlgorithms._smallest_last_ordering(graph)
            elif ordering == "random":
                node_ordering = list(graph.nodes())
                random.shuffle(node_ordering)
            else:
                node_ordering = list(graph.nodes())

            coloring = nx.coloring.greedy_color(graph, strategy=node_ordering)

        elif strategy == "welsh_powell":
            # Welsh-Powell algorithm
            coloring = SpecializedAlgorithms._welsh_powell_coloring(graph)

        elif strategy == "dsatur":
            # DSatur algorithm
            coloring = SpecializedAlgorithms._dsatur_coloring(graph)

        elif strategy == "brooks":
            # Brooks' theorem based coloring
            max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0

            # Check if graph is complete or odd cycle
            if nx.is_complete_graph(graph) or (nx.is_cycle_graph(graph) and graph.number_of_nodes() % 2 == 1):
                # Need max_degree + 1 colors
                coloring = nx.coloring.greedy_color(graph)
            else:
                # Can use max_degree colors (Brooks' theorem)
                coloring = SpecializedAlgorithms._brooks_coloring(graph, max_degree)

        else:
            # Default NetworkX coloring
            coloring = nx.coloring.greedy_color(graph)

        # Calculate statistics
        num_colors = max(coloring.values()) + 1 if coloring else 0

        # Color distribution
        color_counts = defaultdict(int)
        for color in coloring.values():
            color_counts[color] += 1

        # Verify coloring
        is_valid = SpecializedAlgorithms._verify_coloring(graph, coloring)

        # Chromatic number bounds
        CLIQUE_NUMBER_LIMIT = 100  # noqa: PLR2004
        clique_number = nx.graph_clique_number(graph) if graph.number_of_nodes() < CLIQUE_NUMBER_LIMIT else None

        results = {
            "coloring": coloring,
            "num_colors_used": num_colors,
            "color_distribution": dict(color_counts),
            "is_valid_coloring": is_valid,
            "strategy": strategy,
            "lower_bound": clique_number,  # χ(G) ≥ ω(G)
            "upper_bound": max(dict(graph.degree()).values()) + 1 if graph.number_of_nodes() > 0 else 0,
            "execution_time_ms": (time.time() - start_time) * 1000
        }

        return results

    @staticmethod
    def _smallest_last_ordering(graph: nx.Graph) -> List:
        """Smallest-last vertex ordering."""
        G = graph.copy()
        ordering = []

        while G.number_of_nodes() > 0:
            # Find node with minimum degree
            min_node = min(G.nodes(), key=lambda x: G.degree(x))
            ordering.append(min_node)
            G.remove_node(min_node)

        return list(reversed(ordering))

    @staticmethod
    def _welsh_powell_coloring(graph: nx.Graph) -> Dict[Any, int]:
        """Welsh-Powell coloring algorithm."""
        # Sort vertices by degree (descending)
        vertices = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)

        coloring = {}
        color = 0

        while vertices:
            # Color first uncolored vertex
            current_color_nodes = []
            remaining_vertices = []

            for v in vertices:
                # Check if v can be colored with current color
                can_color = True
                for colored in current_color_nodes:
                    if graph.has_edge(v, colored):
                        can_color = False
                        break

                if can_color:
                    coloring[v] = color
                    current_color_nodes.append(v)
                else:
                    remaining_vertices.append(v)

            vertices = remaining_vertices
            color += 1

        return coloring

    @staticmethod
    def _dsatur_coloring(graph: nx.Graph) -> Dict[Any, int]:
        """DSatur (Degree of Saturation) coloring algorithm."""
        coloring = {}
        saturation = dict.fromkeys(graph.nodes(), 0)
        uncolored = set(graph.nodes())

        # Color first vertex (highest degree)
        if uncolored:
            first = max(uncolored, key=lambda x: graph.degree(x))
            coloring[first] = 0
            uncolored.remove(first)

            # Update saturation
            for neighbor in graph[first]:
                if neighbor in uncolored:
                    saturation[neighbor] = 1

        # Color remaining vertices
        while uncolored:
            # Choose vertex with highest saturation, break ties by degree
            next_vertex = max(
                uncolored,
                key=lambda x: (saturation[x], graph.degree(x))
            )

            # Find smallest available color
            neighbor_colors = {
                coloring[n] for n in graph[next_vertex] if n in coloring
            }

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[next_vertex] = color
            uncolored.remove(next_vertex)

            # Update saturation
            for neighbor in graph[next_vertex]:
                if neighbor in uncolored:
                    neighbor_colors_sat = {
                        coloring[n] for n in graph[neighbor] if n in coloring
                    }
                    saturation[neighbor] = len(neighbor_colors_sat)

        return coloring

    @staticmethod
    def _brooks_coloring(graph: nx.Graph, _max_colors: int) -> Dict[Any, int]:
        """Attempt to color with at most max_degree colors (Brooks' theorem)."""
        # This is a simplified version
        # Full Brooks' algorithm is more complex
        return nx.coloring.greedy_color(graph, strategy="saturation_largest_first")

    @staticmethod
    def _verify_coloring(graph: nx.Graph, coloring: Dict[Any, int]) -> bool:
        """Verify that coloring is valid."""
        for u, v in graph.edges():
            if u in coloring and v in coloring:
                if coloring[u] == coloring[v]:
                    return False
        return True

    @staticmethod
    def maximum_clique(
        graph: nx.Graph,
        method: str = "approximation",
        **_params
    ) -> Dict[str, Any]:
        """
        Find maximum clique using exact or approximate algorithms.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        method : str
            Method: 'exact', 'approximation', 'heuristic'

        Returns:
        --------
        Dict containing clique and statistics
        """
        start_time = time.time()

        EXACT_CLIQUE_LIMIT = 50  # noqa: PLR2004
        if method == "exact" and graph.number_of_nodes() < EXACT_CLIQUE_LIMIT:
            # Exact algorithm for small graphs
            max_clique = max(nx.find_cliques(graph), key=len, default=[])
            all_max_cliques = [c for c in nx.find_cliques(graph) if len(c) == len(max_clique)]

            results = {
                "max_clique": list(max_clique),
                "clique_size": len(max_clique),
                "num_max_cliques": len(all_max_cliques),
                "all_max_cliques": all_max_cliques[:10],  # Limit to 10
                "method": "exact"
            }

        elif method == "approximation":
            # Approximation algorithm
            clique = nx.approximation.max_clique(graph)

            results = {
                "max_clique": list(clique),
                "clique_size": len(clique),
                "method": "approximation",
                "guarantee": "2-approximation"
            }

        else:  # heuristic
            # Greedy heuristic
            clique = SpecializedAlgorithms._greedy_clique_heuristic(graph)

            results = {
                "max_clique": list(clique),
                "clique_size": len(clique),
                "method": "greedy_heuristic"
            }

        # Calculate clique number bounds
        # Lower bound: size of found clique
        lower_bound = results["clique_size"]

        # Upper bound: Lovász theta (if scipy available)
        try:
            # Simplified upper bound using max degree
            max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0
            upper_bound = min(max_degree + 1, graph.number_of_nodes())
        except Exception as e:
            logger.debug(f"Failed to compute upper bound for chromatic number: {e}")
            upper_bound = graph.number_of_nodes()

        results.update({
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "graph_density": nx.density(graph),
            "execution_time_ms": (time.time() - start_time) * 1000
        })

        return results

    @staticmethod
    def _greedy_clique_heuristic(graph: nx.Graph) -> Set:
        """Greedy heuristic for finding large clique."""
        # Start with highest degree node
        nodes_by_degree = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)

        best_clique = set()

        for start_node in nodes_by_degree[:10]:  # Try top 10 nodes
            clique = {start_node}
            candidates = set(graph[start_node])

            while candidates:
                # Choose candidate with most connections to current clique
                best_candidate = None
                best_connections = -1

                for candidate in candidates:
                    connections = sum(1 for node in clique if graph.has_edge(candidate, node))
                    if connections == len(clique):  # Connected to all
                        if connections > best_connections:
                            best_connections = connections
                            best_candidate = candidate

                if best_candidate and best_connections == len(clique):
                    clique.add(best_candidate)
                    # Update candidates
                    candidates = candidates.intersection(set(graph[best_candidate]))
                else:
                    break

            if len(clique) > len(best_clique):
                best_clique = clique

        return best_clique

    @staticmethod
    def graph_matching(
        graph: nx.Graph,
        matching_type: str = "maximum",
        weight: Optional[str] = None,
        **_params
    ) -> Dict[str, Any]:
        """
        Find various types of graph matchings.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        matching_type : str
            Type: 'maximum', 'maximal', 'perfect'
        weight : str
            Edge attribute for weighted matching

        Returns:
        --------
        Dict containing matching and statistics
        """
        start_time = time.time()

        if matching_type == "maximum":
            # Maximum cardinality or weight matching
            if weight:
                matching = nx.max_weight_matching(graph, weight=weight)
            else:
                matching = nx.max_weight_matching(graph)

            matching_edges = list(matching)

        elif matching_type == "maximal":
            # Maximal matching (greedy)
            matching = nx.maximal_matching(graph)
            matching_edges = list(matching)

        elif matching_type == "perfect":
            # Check if perfect matching exists
            if graph.number_of_nodes() % 2 != 0:
                return {
                    "has_perfect_matching": False,
                    "reason": "Odd number of nodes",
                    "execution_time_ms": (time.time() - start_time) * 1000
                }

            matching = nx.max_weight_matching(graph)
            matching_edges = list(matching)

            is_perfect = len(matching_edges) * 2 == graph.number_of_nodes()

            if not is_perfect:
                return {
                    "has_perfect_matching": False,
                    "matching_size": len(matching_edges),
                    "uncovered_nodes": graph.number_of_nodes() - len(matching_edges) * 2,
                    "execution_time_ms": (time.time() - start_time) * 1000
                }

        else:
            msg = f"Unknown matching type: {matching_type}"
            raise ValueError(msg)

        # Calculate statistics
        matched_nodes = set()
        for u, v in matching_edges:
            matched_nodes.add(u)
            matched_nodes.add(v)

        total_weight = 0
        if weight:
            for u, v in matching_edges:
                edge_weight = graph[u][v].get(weight, 1)
                total_weight += edge_weight

        results = {
            "matching_type": matching_type,
            "matching_edges": matching_edges,
            "matching_size": len(matching_edges),
            "matched_nodes": list(matched_nodes),
            "num_matched_nodes": len(matched_nodes),
            "coverage": len(matched_nodes) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            "is_perfect": len(matched_nodes) == graph.number_of_nodes(),
            "weighted": weight is not None
        }

        if weight:
            results["total_weight"] = total_weight

        # Maximum possible matching size (Tutte-Berge formula)
        # For general graphs: ν(G) = (n - odd_components + component_count) / 2
        # Simplified: at most n/2
        results["max_possible_matching"] = graph.number_of_nodes() // 2

        results["execution_time_ms"] = (time.time() - start_time) * 1000

        return results

    @staticmethod
    def vertex_cover(
        graph: nx.Graph,
        method: str = "approximation",
        **_params
    ) -> Dict[str, Any]:
        """
        Find minimum vertex cover.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        method : str
            Method: 'approximation', 'ilp' (for small graphs)

        Returns:
        --------
        Dict containing vertex cover
        """
        start_time = time.time()

        ILP_LIMIT = 30  # noqa: PLR2004

        if method == "approximation":
            # 2-approximation algorithm
            cover = nx.approximation.min_weighted_vertex_cover(graph)

            results = {
                "vertex_cover": list(cover),
                "size": len(cover),
                "method": "2-approximation",
                "approximation_ratio": 2.0
            }

        elif method == "ilp" and graph.number_of_nodes() < ILP_LIMIT:
            # Integer Linear Programming (exact for small graphs)
            # Simplified: use matching-based approximation
            matching = nx.max_weight_matching(graph)
            cover = set()
            for u, v in matching:
                cover.add(u)
                cover.add(v)

            # Ensure all edges are covered
            for u, v in graph.edges():
                if u not in cover and v not in cover:
                    cover.add(u)  # Add one endpoint

            results = {
                "vertex_cover": list(cover),
                "size": len(cover),
                "method": "matching_based"
            }

        else:
            # Greedy heuristic
            cover = SpecializedAlgorithms._greedy_vertex_cover(graph)

            results = {
                "vertex_cover": list(cover),
                "size": len(cover),
                "method": "greedy_heuristic"
            }

        # Verify cover
        is_valid = SpecializedAlgorithms._verify_vertex_cover(graph, cover)
        results["is_valid_cover"] = is_valid

        # Lower bound: maximum matching size
        matching_size = len(nx.max_weight_matching(graph))
        results["lower_bound"] = matching_size

        # Cover efficiency
        results["efficiency"] = len(cover) / graph.number_of_edges() if graph.number_of_edges() > 0 else 0

        results["execution_time_ms"] = (time.time() - start_time) * 1000

        return results

    @staticmethod
    def _greedy_vertex_cover(graph: nx.Graph) -> Set:
        """Greedy vertex cover heuristic."""
        G = graph.copy()
        cover = set()

        while G.number_of_edges() > 0:
            # Choose vertex with highest degree
            v = max(G.nodes(), key=lambda x: G.degree(x))
            cover.add(v)
            G.remove_node(v)

        return cover

    @staticmethod
    def _verify_vertex_cover(graph: nx.Graph, cover: Set) -> bool:
        """Verify that cover covers all edges."""
        for u, v in graph.edges():
            if u not in cover and v not in cover:
                return False
        return True

    @staticmethod
    def dominating_set(
        graph: nx.Graph,
        method: str = "greedy",
        **_params
    ) -> Dict[str, Any]:
        """
        Find minimum dominating set.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        method : str
            Method: 'greedy', 'approximation'

        Returns:
        --------
        Dict containing dominating set
        """
        start_time = time.time()

        if method == "greedy":
            dom_set = SpecializedAlgorithms._greedy_dominating_set(graph)

        elif method == "approximation":
            # Use approximation algorithm
            dom_set = nx.approximation.min_weighted_dominating_set(graph)

        else:
            dom_set = SpecializedAlgorithms._greedy_dominating_set(graph)

        # Calculate statistics
        dominated_nodes = set(dom_set)
        for node in dom_set:
            dominated_nodes.update(graph[node])

        results = {
            "dominating_set": list(dom_set),
            "size": len(dom_set),
            "dominated_nodes": list(dominated_nodes),
            "is_total_dominating": len(dominated_nodes) == graph.number_of_nodes(),
            "method": method,
            "execution_time_ms": (time.time() - start_time) * 1000
        }

        # Lower bound
        # Every node can dominate at most degree + 1 nodes
        max_domination = max(graph.degree(n) + 1 for n in graph.nodes()) if graph.number_of_nodes() > 0 else 1
        results["lower_bound"] = graph.number_of_nodes() / max_domination

        return results

    @staticmethod
    def _greedy_dominating_set(graph: nx.Graph) -> Set:
        """Greedy dominating set heuristic."""
        dominated = set()
        dom_set = set()

        while len(dominated) < graph.number_of_nodes():
            # Choose node that dominates most undominated nodes
            best_node = None
            best_count = -1

            for node in graph.nodes():
                if node not in dom_set:
                    # Count undominated neighbors
                    count = 0
                    if node not in dominated:
                        count += 1
                    for neighbor in graph[node]:
                        if neighbor not in dominated:
                            count += 1

                    if count > best_count:
                        best_count = count
                        best_node = node

            if best_node:
                dom_set.add(best_node)
                dominated.add(best_node)
                dominated.update(graph[best_node])
            else:
                break

        return dom_set

    @staticmethod
    def link_prediction(
        graph: nx.Graph,
        method: str = "common_neighbors",
        node_pairs: Optional[List[Tuple]] = None,
        top_k: int = 10,
        **_params
    ) -> Dict[str, Any]:
        """
        Predict missing links in the graph.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        method : str
            Method: 'common_neighbors', 'adamic_adar', 'jaccard', 'preferential_attachment'
        node_pairs : List[Tuple]
            Specific pairs to evaluate (default: all non-edges)
        top_k : int
            Return top k predictions

        Returns:
        --------
        Dict containing link predictions
        """
        start_time = time.time()

        # Get node pairs to evaluate
        if node_pairs is None:
            # Sample non-edges for large graphs
            LINK_PREDICTION_SAMPLE_LIMIT = 100  # noqa: PLR2004
            if graph.number_of_nodes() > LINK_PREDICTION_SAMPLE_LIMIT:
                all_possible = graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2
                existing = graph.number_of_edges()
                non_edges = all_possible - existing

                # Sample at most 1000 non-edges
                sample_size = min(1000, non_edges)
                node_pairs = []

                nodes = list(graph.nodes())
                attempts = 0
                while len(node_pairs) < sample_size and attempts < sample_size * 3:
                    u = random.choice(nodes)  # noqa: S311
                    v = random.choice(nodes)  # noqa: S311
                    if u != v and not graph.has_edge(u, v) and (u, v) not in node_pairs and (v, u) not in node_pairs:
                        node_pairs.append((u, v))
                    attempts += 1
            else:
                # All non-edges for small graphs
                node_pairs = []
                nodes = list(graph.nodes())
                for i, u in enumerate(nodes):
                    for v in nodes[i+1:]:
                        if not graph.has_edge(u, v):
                            node_pairs.append((u, v))

        # Calculate scores
        scores = []

        if method == "common_neighbors":
            for u, v in node_pairs:
                score = len(list(nx.common_neighbors(graph, u, v)))
                scores.append(((u, v), score))

        elif method == "adamic_adar":
            for u, v in node_pairs:
                score = nx.adamic_adar_index(graph, [(u, v)]).__next__()[2]
                scores.append(((u, v), score))

        elif method == "jaccard":
            for u, v in node_pairs:
                score = nx.jaccard_coefficient(graph, [(u, v)]).__next__()[2]
                scores.append(((u, v), score))

        elif method == "preferential_attachment":
            for u, v in node_pairs:
                score = nx.preferential_attachment(graph, [(u, v)]).__next__()[2]
                scores.append(((u, v), score))

        elif method == "resource_allocation":
            for u, v in node_pairs:
                score = nx.resource_allocation_index(graph, [(u, v)]).__next__()[2]
                scores.append(((u, v), score))

        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Get top k
        top_predictions = scores[:top_k]

        results = {
            "method": method,
            "top_predictions": [
                {
                    "node_pair": pair,
                    "score": score,
                    "rank": i + 1
                }
                for i, (pair, score) in enumerate(top_predictions)
            ],
            "num_candidates_evaluated": len(node_pairs),
            "score_distribution": {
                "min": min(s[1] for s in scores) if scores else 0,
                "max": max(s[1] for s in scores) if scores else 0,
                "mean": np.mean([s[1] for s in scores]) if scores else 0,
                "std": np.std([s[1] for s in scores]) if scores else 0
            },
            "execution_time_ms": (time.time() - start_time) * 1000
        }

        return results
