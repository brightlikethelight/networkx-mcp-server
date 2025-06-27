"""Bipartite graph analysis and algorithms."""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from networkx.algorithms import bipartite

logger = logging.getLogger(__name__)


class BipartiteAnalysis:
    """Analysis tools for bipartite graphs."""

    @staticmethod
    def is_bipartite(
        graph: nx.Graph,
        return_sets: bool = True
    ) -> Union[bool, Tuple[bool, Optional[Tuple[Set, Set]]]]:
        """
        Check if graph is bipartite and optionally return the bipartition.
        
        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        return_sets : bool
            Whether to return the two sets of nodes
            
        Returns:
        --------
        bool or Tuple[bool, Optional[Tuple[Set, Set]]]
        """
        try:
            is_bip = bipartite.is_bipartite(graph)

            if not return_sets:
                return is_bip

            if is_bip:
                # Get the bipartition
                color_dict = bipartite.color(graph)
                set_0 = {n for n, c in color_dict.items() if c == 0}
                set_1 = {n for n, c in color_dict.items() if c == 1}
                return True, (set_0, set_1)
            else:
                return False, None

        except nx.NetworkXError:
            if return_sets:
                return False, None
            return False

    @staticmethod
    def bipartite_projection(
        graph: nx.Graph,
        nodes: Set[Any],
        weight_function: Optional[str] = "overlap",
        **params
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Create weighted or unweighted projection onto one node set.
        
        Parameters:
        -----------
        graph : nx.Graph
            Bipartite graph
        nodes : Set
            Nodes to project onto
        weight_function : str
            Weight function: 'overlap', 'jaccard', 'newman', None
            
        Returns:
        --------
        Tuple of (projection graph, metadata)
        """
        start_time = time.time()

        # Verify bipartite
        is_bip, sets = BipartiteAnalysis.is_bipartite(graph, return_sets=True)
        if not is_bip:
            raise ValueError("Graph is not bipartite")

        # Determine which set contains the nodes
        if nodes.issubset(sets[0]):
            bottom_nodes = sets[0]
            top_nodes = sets[1]
        elif nodes.issubset(sets[1]):
            bottom_nodes = sets[1]
            top_nodes = sets[0]
        else:
            raise ValueError("Nodes must all belong to one partition")

        # Create projection based on weight function
        if weight_function is None:
            # Unweighted projection
            P = bipartite.projected_graph(graph, nodes)
            edge_weights = None

        elif weight_function == "overlap":
            # Overlap weighted projection
            P = bipartite.overlap_weighted_projected_graph(graph, nodes)
            edge_weights = nx.get_edge_attributes(P, "weight")

        elif weight_function == "jaccard":
            # Custom Jaccard weighted projection
            P = nx.Graph()
            P.add_nodes_from(nodes)

            # For each pair of nodes in the same set
            for u in nodes:
                for v in nodes:
                    if u < v:  # Avoid duplicates
                        # Find common neighbors
                        u_neighbors = set(graph[u])
                        v_neighbors = set(graph[v])

                        intersection = u_neighbors & v_neighbors
                        union = u_neighbors | v_neighbors

                        if len(union) > 0:
                            weight = len(intersection) / len(union)
                            if weight > 0:
                                P.add_edge(u, v, weight=weight)

            edge_weights = nx.get_edge_attributes(P, "weight")

        elif weight_function == "newman":
            # Newman's collaboration weighted projection
            P = bipartite.collaboration_weighted_projected_graph(graph, nodes)
            edge_weights = nx.get_edge_attributes(P, "weight")

        else:
            raise ValueError(f"Unknown weight function: {weight_function}")

        execution_time = (time.time() - start_time) * 1000

        # Calculate projection statistics
        metadata = {
            "projection_nodes": len(P),
            "projection_edges": P.number_of_edges(),
            "original_nodes": graph.number_of_nodes(),
            "original_edges": graph.number_of_edges(),
            "bottom_nodes": len(bottom_nodes),
            "top_nodes": len(top_nodes),
            "weight_function": weight_function,
            "execution_time_ms": execution_time
        }

        if edge_weights:
            weights = list(edge_weights.values())
            metadata.update({
                "min_weight": min(weights),
                "max_weight": max(weights),
                "mean_weight": np.mean(weights),
                "std_weight": np.std(weights)
            })

        # Density comparison
        max_possible_edges = len(nodes) * (len(nodes) - 1) // 2
        metadata["projection_density"] = (
            P.number_of_edges() / max_possible_edges if max_possible_edges > 0 else 0
        )

        return P, metadata

    @staticmethod
    def bipartite_clustering(
        graph: nx.Graph,
        nodes: Optional[Set[Any]] = None,
        mode: str = "dot",
        **params
    ) -> Dict[str, Any]:
        """
        Calculate bipartite clustering coefficients.
        
        Parameters:
        -----------
        graph : nx.Graph
            Bipartite graph
        nodes : Set, optional
            Nodes to calculate clustering for (default: all)
        mode : str
            Clustering mode: 'dot', 'min', 'max'
            
        Returns:
        --------
        Dict containing clustering coefficients and statistics
        """
        # Verify bipartite
        is_bip, sets = BipartiteAnalysis.is_bipartite(graph, return_sets=True)
        if not is_bip:
            raise ValueError("Graph is not bipartite")

        if nodes is None:
            # Calculate for both sets
            results = {}

            for i, node_set in enumerate(sets):
                if mode == "dot":
                    clustering = bipartite.clustering(graph, node_set)
                elif mode == "min":
                    clustering = bipartite.clustering(graph, node_set, mode="min")
                elif mode == "max":
                    clustering = bipartite.clustering(graph, node_set, mode="max")
                else:
                    raise ValueError(f"Unknown clustering mode: {mode}")

                # Calculate statistics
                values = list(clustering.values())
                results[f"set_{i}"] = {
                    "clustering_coefficients": clustering,
                    "average_clustering": np.mean(values) if values else 0,
                    "min_clustering": min(values) if values else 0,
                    "max_clustering": max(values) if values else 0,
                    "std_clustering": np.std(values) if values else 0,
                    "num_nodes": len(node_set)
                }

            # Overall statistics
            all_values = []
            for set_result in results.values():
                all_values.extend(set_result["clustering_coefficients"].values())

            return {
                "by_set": results,
                "overall_average": np.mean(all_values) if all_values else 0,
                "mode": mode,
                "bipartite_sets": [list(s) for s in sets]
            }

        else:
            # Calculate for specific nodes
            if mode == "dot":
                clustering = bipartite.clustering(graph, nodes)
            elif mode == "min":
                clustering = bipartite.clustering(graph, nodes, mode="min")
            elif mode == "max":
                clustering = bipartite.clustering(graph, nodes, mode="max")
            else:
                raise ValueError(f"Unknown clustering mode: {mode}")

            values = list(clustering.values())

            return {
                "clustering_coefficients": clustering,
                "average_clustering": np.mean(values) if values else 0,
                "min_clustering": min(values) if values else 0,
                "max_clustering": max(values) if values else 0,
                "std_clustering": np.std(values) if values else 0,
                "num_nodes": len(nodes),
                "mode": mode
            }

    @staticmethod
    def bipartite_centrality(
        graph: nx.Graph,
        centrality_type: str = "degree",
        normalized: bool = True,
        **params
    ) -> Dict[str, Any]:
        """
        Calculate adapted centrality measures for bipartite graphs.
        
        Parameters:
        -----------
        graph : nx.Graph
            Bipartite graph
        centrality_type : str
            Type: 'degree', 'betweenness', 'closeness'
        normalized : bool
            Whether to normalize values
            
        Returns:
        --------
        Dict containing centrality measures by node set
        """
        # Verify bipartite
        is_bip, sets = BipartiteAnalysis.is_bipartite(graph, return_sets=True)
        if not is_bip:
            raise ValueError("Graph is not bipartite")

        results = {}

        if centrality_type == "degree":
            # Bipartite degree centrality
            top_nodes, bottom_nodes = sets

            # Degree centrality for bottom nodes
            bottom_deg_centrality = bipartite.degree_centrality(graph, bottom_nodes)

            # Degree centrality for top nodes
            top_deg_centrality = bipartite.degree_centrality(graph, top_nodes)

            results = {
                "bottom_nodes": {
                    "centrality": bottom_deg_centrality,
                    "average": np.mean(list(bottom_deg_centrality.values())),
                    "max": max(bottom_deg_centrality.values()),
                    "node_count": len(bottom_nodes)
                },
                "top_nodes": {
                    "centrality": top_deg_centrality,
                    "average": np.mean(list(top_deg_centrality.values())),
                    "max": max(top_deg_centrality.values()),
                    "node_count": len(top_nodes)
                }
            }

        elif centrality_type == "betweenness":
            # Betweenness centrality
            if normalized:
                centrality = bipartite.betweenness_centrality(graph, sets[1])
            else:
                # Manual calculation without normalization
                centrality = nx.betweenness_centrality(graph, normalized=False)

            # Separate by sets
            for i, node_set in enumerate(sets):
                set_centrality = {n: centrality[n] for n in node_set if n in centrality}
                values = list(set_centrality.values())

                results[f"set_{i}"] = {
                    "centrality": set_centrality,
                    "average": np.mean(values) if values else 0,
                    "max": max(values) if values else 0,
                    "min": min(values) if values else 0,
                    "node_count": len(node_set)
                }

        elif centrality_type == "closeness":
            # Closeness centrality
            if not nx.is_connected(graph):
                # Use harmonic centrality for disconnected graphs
                centrality = nx.harmonic_centrality(graph)
                centrality_name = "harmonic_centrality"
            else:
                centrality = bipartite.closeness_centrality(graph, sets[1], normalized=normalized)
                centrality_name = "closeness_centrality"

            # Separate by sets
            for i, node_set in enumerate(sets):
                set_centrality = {n: centrality.get(n, 0) for n in node_set}
                values = list(set_centrality.values())

                results[f"set_{i}"] = {
                    centrality_name: set_centrality,
                    "average": np.mean(values) if values else 0,
                    "max": max(values) if values else 0,
                    "min": min(values) if values else 0,
                    "node_count": len(node_set)
                }

        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")

        return {
            "centrality_type": centrality_type,
            "normalized": normalized,
            "results": results,
            "bipartite_sets": [list(s) for s in sets]
        }

    @staticmethod
    def maximum_matching(
        graph: nx.Graph,
        weight: Optional[str] = None,
        top_nodes: Optional[Set[Any]] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Find maximum matching using Hungarian algorithm.
        
        Parameters:
        -----------
        graph : nx.Graph
            Bipartite graph
        weight : str, optional
            Edge attribute for weights
        top_nodes : Set, optional
            One set of the bipartition
            
        Returns:
        --------
        Dict containing matching and statistics
        """
        start_time = time.time()

        # Verify bipartite
        is_bip, sets = BipartiteAnalysis.is_bipartite(graph, return_sets=True)
        if not is_bip:
            raise ValueError("Graph is not bipartite")

        # Use provided top_nodes or infer
        if top_nodes is None:
            top_nodes = sets[0]

        # Find matching
        if weight is None:
            # Maximum cardinality matching
            matching = bipartite.maximum_matching(graph, top_nodes)
            matching_weight = len(matching) // 2  # Each edge counted twice

        else:
            # Maximum weight matching
            # Create weight dict if needed
            if isinstance(weight, str):
                weights = nx.get_edge_attributes(graph, weight)
            else:
                weights = weight

            # Use minimum weight matching with negative weights
            neg_weights = {e: -w for e, w in weights.items()}

            # NetworkX's max_weight_matching works for general graphs
            # For bipartite, we use it directly
            matching_edges = nx.max_weight_matching(graph, weight=weight)
            matching = {}

            for u, v in matching_edges:
                matching[u] = v
                matching[v] = u

            # Calculate total weight
            matching_weight = sum(
                weights.get((u, v), weights.get((v, u), 0))
                for u, v in matching_edges
            )

        execution_time = (time.time() - start_time) * 1000

        # Calculate statistics
        matched_nodes = set(matching.keys())
        unmatched_nodes = set(graph.nodes()) - matched_nodes

        # Separate by sets
        set0_matched = len(matched_nodes & sets[0])
        set1_matched = len(matched_nodes & sets[1])

        # Perfect matching check
        is_perfect = len(matched_nodes) == graph.number_of_nodes()

        # Maximum possible matching
        max_possible = min(len(sets[0]), len(sets[1]))

        return {
            "matching": dict(matching),
            "matching_size": len(matching) // 2,
            "matching_weight": matching_weight,
            "matched_nodes": list(matched_nodes),
            "unmatched_nodes": list(unmatched_nodes),
            "is_perfect_matching": is_perfect,
            "is_maximum_matching": (len(matching) // 2) == max_possible,
            "set0_matched": set0_matched,
            "set1_matched": set1_matched,
            "coverage": len(matched_nodes) / graph.number_of_nodes(),
            "execution_time_ms": execution_time,
            "weighted": weight is not None
        }

    @staticmethod
    def bipartite_community_detection(
        graph: nx.Graph,
        method: str = "label_propagation",
        **params
    ) -> Dict[str, Any]:
        """
        Detect communities in bipartite graphs using specialized algorithms.
        
        Parameters:
        -----------
        graph : nx.Graph
            Bipartite graph
        method : str
            Method: 'label_propagation', 'modularity', 'barber'
            
        Returns:
        --------
        Dict containing communities and metadata
        """
        start_time = time.time()

        # Verify bipartite
        is_bip, sets = BipartiteAnalysis.is_bipartite(graph, return_sets=True)
        if not is_bip:
            raise ValueError("Graph is not bipartite")

        if method == "label_propagation":
            # Bipartite-aware label propagation
            communities = BipartiteAnalysis._bipartite_label_propagation(
                graph, sets[0], sets[1], **params
            )

        elif method == "modularity":
            # Project and detect communities
            # Project onto larger set
            larger_set = sets[0] if len(sets[0]) >= len(sets[1]) else sets[1]
            projection, _ = BipartiteAnalysis.bipartite_projection(
                graph, larger_set, weight_function="overlap"
            )

            # Detect communities in projection
            from networkx.algorithms.community import greedy_modularity_communities
            proj_communities = list(greedy_modularity_communities(projection))

            # Map back to original nodes
            communities = []
            for comm in proj_communities:
                # Include nodes from both sets
                full_comm = set(comm)

                # Add connected nodes from other set
                for node in comm:
                    full_comm.update(graph[node])

                communities.append(full_comm)

        elif method == "barber":
            # Barber's bipartite modularity optimization
            communities = BipartiteAnalysis._barber_modularity(graph, sets[0], sets[1], **params)

        else:
            raise ValueError(f"Unknown method: {method}")

        execution_time = (time.time() - start_time) * 1000

        # Calculate bipartite modularity
        modularity = BipartiteAnalysis._calculate_bipartite_modularity(
            graph, communities, sets[0], sets[1]
        )

        # Community statistics
        comm_sizes = [len(c) for c in communities]

        # Check how communities span both sets
        spanning_stats = []
        for comm in communities:
            set0_nodes = len(comm & sets[0])
            set1_nodes = len(comm & sets[1])
            spanning_stats.append({
                "total": len(comm),
                "set0": set0_nodes,
                "set1": set1_nodes,
                "balance": min(set0_nodes, set1_nodes) / max(set0_nodes, set1_nodes, 1)
            })

        return {
            "communities": [list(c) for c in communities],
            "num_communities": len(communities),
            "modularity": modularity,
            "community_sizes": comm_sizes,
            "spanning_statistics": spanning_stats,
            "method": method,
            "execution_time_ms": execution_time
        }

    @staticmethod
    def _bipartite_label_propagation(
        graph: nx.Graph,
        set0: Set,
        set1: Set,
        max_iterations: int = 100,
        **params
    ) -> List[Set]:
        """Bipartite-aware label propagation."""
        # Initialize each node with its own label
        labels = {node: i for i, node in enumerate(graph.nodes())}

        for iteration in range(max_iterations):
            changed = False

            # Update labels for set 0 based on set 1 neighbors
            for node in set0:
                neighbor_labels = [labels[n] for n in graph[node] if n in set1]
                if neighbor_labels:
                    # Most frequent label among neighbors
                    label_counts = defaultdict(int)
                    for label in neighbor_labels:
                        label_counts[label] += 1

                    new_label = max(label_counts, key=label_counts.get)
                    if labels[node] != new_label:
                        labels[node] = new_label
                        changed = True

            # Update labels for set 1 based on set 0 neighbors
            for node in set1:
                neighbor_labels = [labels[n] for n in graph[node] if n in set0]
                if neighbor_labels:
                    label_counts = defaultdict(int)
                    for label in neighbor_labels:
                        label_counts[label] += 1

                    new_label = max(label_counts, key=label_counts.get)
                    if labels[node] != new_label:
                        labels[node] = new_label
                        changed = True

            if not changed:
                break

        # Convert labels to communities
        communities_dict = defaultdict(set)
        for node, label in labels.items():
            communities_dict[label].add(node)

        return list(communities_dict.values())

    @staticmethod
    def _barber_modularity(
        graph: nx.Graph,
        set0: Set,
        set1: Set,
        resolution: float = 1.0,
        **params
    ) -> List[Set]:
        """Barber's bipartite modularity optimization (simplified)."""
        # This is a simplified greedy approach
        # Full implementation would use more sophisticated optimization

        # Start with each node in its own community
        communities = [{node} for node in graph.nodes()]

        # Greedily merge communities
        improved = True
        while improved:
            improved = False
            best_merge = None
            best_delta = 0

            # Try all pairs of communities
            for i in range(len(communities)):
                for j in range(i + 1, len(communities)):
                    # Calculate modularity change if merged
                    merged = communities[i] | communities[j]

                    # Current modularity
                    current_mod = BipartiteAnalysis._calculate_bipartite_modularity(
                        graph, communities, set0, set1
                    )

                    # New modularity
                    new_communities = [c for k, c in enumerate(communities) if k != i and k != j]
                    new_communities.append(merged)
                    new_mod = BipartiteAnalysis._calculate_bipartite_modularity(
                        graph, new_communities, set0, set1
                    )

                    delta = new_mod - current_mod

                    if delta > best_delta:
                        best_delta = delta
                        best_merge = (i, j)
                        improved = True

            if improved and best_merge:
                i, j = best_merge
                # Merge communities
                merged = communities[i] | communities[j]
                communities = [c for k, c in enumerate(communities) if k != i and k != j]
                communities.append(merged)

        return communities

    @staticmethod
    def _calculate_bipartite_modularity(
        graph: nx.Graph,
        communities: List[Set],
        set0: Set,
        set1: Set
    ) -> float:
        """Calculate Barber's bipartite modularity."""
        m = graph.number_of_edges()
        if m == 0:
            return 0

        # Degree sequences
        degrees0 = {n: graph.degree(n) for n in set0}
        degrees1 = {n: graph.degree(n) for n in set1}

        modularity = 0

        for community in communities:
            comm0 = community & set0
            comm1 = community & set1

            # Actual edges within community (between sets)
            actual_edges = 0
            for u in comm0:
                for v in graph[u]:
                    if v in comm1:
                        actual_edges += 1

            # Expected edges
            expected_edges = 0
            for u in comm0:
                for v in comm1:
                    expected_edges += degrees0[u] * degrees1[v] / (2 * m)

            modularity += (actual_edges - expected_edges)

        return modularity / m
